from typing import Dict, List
import re
import csv
import json
from datetime import datetime


import torch
import requests
from sklearn import metrics
from django.contrib import messages
from django.shortcuts import render, redirect
from django.http import JsonResponse, StreamingHttpResponse
from django.core.cache import cache
from django.db.models import F
from django.contrib.postgres.aggregates import ArrayAgg
from django.core.serializers import serialize
from django.views.decorators.cache import cache_page
from infer.forms import ProteomeSubmitForm, GOEnrichmentForm, DiagnosticForm
from infer.tasks.infer import pairwise_predict, get_model, infer_batch
from infer.models import Protein, Weights, Organism, ProteomeTask, ProteomeResult, get_all_task_result, get_proteome_stats, \
    get_proteome_probability_histogram, get_proteome_go_frequency, Architecture, ProteinEmbedding
from infer.utils import generate_token, valid_token, hash_dict
from django_q.tasks import async_task, result


def clean_sequence(seq):
    aa_pattern = re.compile("^[ARNDCQEGHILKMFPSTWYVOUBZX]+$")
    seq = seq.upper()
    seq = re.sub(r"\s", "", seq)

    if aa_pattern.fullmatch(seq) is None:
        return None
    else:
        return seq


@cache_page(60 * 60)
def infer_index(request):
    return render(request, "infer/index.html")


def print_result(task_id):
    print(result(task_id))


def proteome_submit(request):
    if request.method == "POST":
        form = ProteomeSubmitForm(request.POST)
        if form.is_valid():
            result_id = generate_token("PROTEO1")

            taxon = form.cleaned_data["organism"]

            try:
                organism = Organism.objects.get(pk=taxon)
            except Organism.DoesNotExist:
                messages.add_message(request, messages.ERROR, 'Invalid Organism selected.')
                return redirect("proteome_submit")

            seq = clean_sequence(form.cleaned_data["seq"])

            if seq is None:
                messages.add_message(request, messages.ERROR, 'Invalid Amino Acid sequence.')
                return redirect("proteome_submit")

            #weights = Weights.objects.get(name="nest-much")
            try:
                weights = Weights.objects.filter(architecture_id=form.cleaned_data["architecture"], enabled=True).latest(
                    'pub_date')
            except Weights.DoesNotExist:
                messages.add_message(request, messages.ERROR, 'Could find no model weights for the requested architecture.')
                return redirect("proteome_submit")

            # async_task(compute_embeddings, weights, [seq])
            task_id = async_task(pairwise_predict, result_id, weights, seq, organism)

            results = ProteomeTask(
                id=result_id,
                status="Q",
                qid=task_id,
                organism=organism,
                sequence=seq[:3000],
                weights=weights,
                submission_timestamp=datetime.now(),
            )
            results.save()

            return redirect("proteome_report", result_id=result_id)
        else:
            return redirect("proteome_submit")
    else:
        form = ProteomeSubmitForm()

    return render(request, "infer/submit.html", {"form": form})


def task_dict(result_id) -> dict:

    cache_result = cache.get(f"PROTEOME_TASK_DICT_{result_id}")

    # if cache_result:
    #    return cache_result

    try:
        proteome_task = ProteomeTask.objects.get(id=result_id)
    except ProteomeTask.DoesNotExist:
        raise ValueError(f"Can't make task_dict, no ProteomeTask with ID {result_id}")

    # We'll whitelist the fields we want here to avoid leaking any data that can result in abuse

    def whitelist_fields(fields: Dict, whitelist: List[str]):
        return {key: fields[key] for key in fields if key in whitelist}

    proteome_task_dict = serialize("python", [proteome_task])[0]["fields"]

    proteome_task_dict["submission_timestamp"] = proteome_task_dict[
        "submission_timestamp"
    ].timestamp()

    if proteome_task_dict["completion_timestamp"]:
        proteome_task_dict["completion_timestamp"] = proteome_task_dict[
            "completion_timestamp"
        ].timestamp()
    else:
        proteome_task_dict["completion_timestamp"] = None

    organism_dict_unsafe = serialize("python", [proteome_task.organism])[0]["fields"]
    organism_whitelist = ["ncbi_taxon", "common_name", "scientific_name"]
    organism_dict = whitelist_fields(organism_dict_unsafe, organism_whitelist)
    proteome_task_dict["organism"] = organism_dict

    weights_dict_unsafe = serialize("python", [proteome_task.weights])[0]["fields"]
    weights_whitelist = ["name", "pub_date", "trunc_len"]
    weights_dict = whitelist_fields(weights_dict_unsafe, weights_whitelist)
    weights_dict["pub_date"] = weights_dict["pub_date"].timestamp()
    proteome_task_dict["weights"] = weights_dict

    weights_dataset_dict_unsafe = serialize("python", [proteome_task.weights.dataset])[
        0
    ]["fields"]
    weights_dataset_whitelist = ["name", "pub_date", "trunc_len"]
    weights_dataset_dict = whitelist_fields(
        weights_dataset_dict_unsafe, weights_dataset_whitelist
    )
    proteome_task_dict["weights"]["dataset"] = weights_dataset_dict

    weights_dataset_ppi_db_dict_unsafe = serialize(
        "python", [proteome_task.weights.dataset.ppi_database]
    )[0]["fields"]
    weights_dataset_ppi_db_whitelist = ["name", "version", "link"]
    weights_dataset_ppi_db_dict = whitelist_fields(
        weights_dataset_ppi_db_dict_unsafe, weights_dataset_ppi_db_whitelist
    )
    proteome_task_dict["weights"]["dataset"][
        "ppi_database"
    ] = weights_dataset_ppi_db_dict

    weights_dataset_organisms_dict = serialize(
        "python", proteome_task.weights.dataset.organisms.all()
    )
    weights_dataset_organisms_dict = [
        whitelist_fields(o["fields"], organism_whitelist)
        for o in weights_dataset_organisms_dict
    ]
    proteome_task_dict["weights"]["dataset"][
        "organisms"
    ] = weights_dataset_organisms_dict

    proteome_task_dict["status"] = "D"

    cache.set(f"PROTEOME_TASK_DICT_{result_id}", proteome_task_dict)

    return proteome_task_dict


def get_num_results_proteome(result_id: str) -> int:

    cache_key = f"PROTEOME_RESULTS_TOTAL_{result_id}"
    cache_result = cache.get(cache_key)

    if cache_result:
        return cache_result
    else:
        count = ProteomeResult.objects.filter(task_id=result_id).count()
        cache.set(cache_key, count, timeout=3600)
        return count


def proteome_task_json(request, result_id):
    """
    Returns progress of a proteome task in JSON format.

    There's always a status key returned, which is always one character long.

        - D: The task is done. (Status 200)
        - E: An internal error has occured in displaying the progress. The task might have completed just fine, though. (Status 500)
        - P: The task is in progress. The keys total_proteins and computed_proteins might be defined. If they are not, it's possible the task is interminably hung. (Status 200)
        - Q: The task is sitting in the Queue.
        - F: The task has failed.
        - ?: Unknown task.

    :param result_id: The ProteomeTask ID
    """

    if not valid_token("PROTEO1", result_id):
        return render(request, "status/404.html", status=404)

    result = cache.get(f"STATUS-{result_id}")

    if result:
        result = json.loads(result)

        if (
            "total_proteins" not in result
            or "computed_proteins" not in result
            or "status" not in result
        ):
            return JsonResponse({"status": "E"}, status=500)

        if result["total_proteins"] == result["computed_proteins"]:
            cache.delete(f"STATUS-{result_id}")
            return JsonResponse(task_dict(result_id), status=200)  # D

        return JsonResponse(result)
    else:
        try:
            proteome_task = ProteomeTask.objects.get(id=result_id)

            if proteome_task.status == "D":
                return JsonResponse(task_dict(result_id), status=200)
            elif proteome_task.status == "E":
                status = 500
            elif proteome_task.status == "P":
                status = 202
            elif proteome_task.status == "F":
                status = 500
            elif proteome_task.status == "Q":
                status = 202
            else:
                raise ValueError("Unexpected ProteomeTask Status Code")

            return JsonResponse({"status": proteome_task.status}, status=status)

        except ProteomeTask.DoesNotExist:
            return JsonResponse({"status": "?"}, status=404)


def proteome_results_json(request, result_id):

    if not valid_token("PROTEO1", result_id):
        return render(request, "status/404.html", status=404)

    query = request.GET.dict()

    # DataTables adds a random "_" and a monotonicly increase "draw" key which will interfere with our caching
    deterministic_query = query.copy()
    if "_" in deterministic_query:
        del deterministic_query["_"]
    if "draw" in deterministic_query:
        del deterministic_query["draw"]

    cache_key = json.dumps(
        hash_dict(
            {
                "namespace": "ProteomeResult",
                "task_id": result_id,
                "query": deterministic_query,
            }
        )
    )

    print("!!!!!! ", cache_key)
    print("###### ", deterministic_query)

    result_dict = cache.get(cache_key)

    if result_dict is not None:
        # Update the draw key which is out of date.
        result_dict["draw"] = query["draw"]
        return JsonResponse(result_dict, safe=False)

    # put an upper bound on the number of query keys we can receive.
    # this is to prevent slowing down things when we search the query dict.
    if len(query) > 200:
        return JsonResponse({"error": "too many parameters"}, status=400)

    if "draw" not in query:
        return JsonResponse({"error": "draw parameter not provided"}, status=400)

    if "start" not in query:
        query["start"] = 0

    if "length" not in query:
        query["length"] = 50

    for int_key in ["draw", "start", "length", "order"]:
        try:
            if int_key in query:
                query[int_key] = int(query[int_key])
        except ValueError:
            return JsonResponse(
                {"error": f"invalid parameter for {int_key}, must be integer"},
                status=400,
            )

    if query["length"] > 100:
        query["length"] = 100

    if query["length"] == -1:
        query["length"] = 100

    start = query["start"]
    end = start + query["length"]

    if start > end or start < 0:
        return JsonResponse(
            {"error": "combination of start and length parameters is invalid."},
            status=400,
        )

    # First, lets do some validation on the ProteomeTask status/existence.
    # This is important because we're going to cache our result, so incomplete returns are going to mess things up.
    try:
        proteome_task = ProteomeTask.objects.get(id=result_id)
        if proteome_task.status == "P":
            return JsonResponse({"status": "P"}, status=202)
        elif proteome_task.status == "Q":
            return JsonResponse({"status": "Q"}, status=202)
        elif proteome_task.status == "E":
            return JsonResponse({"status": "E"}, status=500)
        elif proteome_task.status == "F":
            return JsonResponse({"status": "F"}, status=500)
        elif proteome_task.status != "D":
            return JsonResponse({"status": "E"}, status=500)

    except ProteomeTask.DoesNotExist:
        return JsonResponse({"status": "?"}, status=404)

    if "search[value]" in query and query["search[value]"] != "":
        search_term = query["search[value]"]

        results_uniprot_ac = (
            ProteomeResult.objects.filter(task_id=result_id, protein__reviewed=True, protein__id__iexact=search_term)
            .exclude(protein__gene_name="")
            .order_by("-probability")
            .all()
        )

        results_gene_name = (
            ProteomeResult.objects.filter(task_id=result_id, protein__reviewed=True,
                                          protein__gene_name__icontains=search_term)
            .exclude(protein__gene_name="")
            .order_by("-probability")
            .all()
        )

        results_protein_name = (
            ProteomeResult.objects.filter(task_id=result_id, protein__reviewed=True,
                                          protein__protein_name__icontains=search_term)
            .exclude(protein__gene_name="")
            .order_by("-probability")
            .all()
        )

        results = results_uniprot_ac | results_gene_name | results_protein_name
        results = results[start:end].annotate(
                uniprot_ac=F("protein_id"),
                gene_name=F("protein__gene_name"),
                protein_name=F("protein__protein_name"),
                protein_go_id=ArrayAgg("protein__gene_ontologies")).values(
                "protein", "uniprot_ac", "probability", "gene_name", "protein_name", "protein_go_id"
            )
    else:
        results = get_all_task_result(result_id)[start:end]

    total_records = get_num_results_proteome(result_id)

    result_dict = {"draw": query["draw"], "recordsTotal": total_records, "recordsFiltered": total_records, "data": []}

    for idx, result in enumerate(results):
        protein_id = result["protein"]
        probability = result["probability"]
        gene_name = result["gene_name"]
        protein_name = result["protein_name"]
        protein_go_id = result["protein_go_id"]

        row = {
            "uniprot_ac": protein_id,
            "gene_name": gene_name,
            "protein_name": protein_name,
            "probability": probability,
            "protein_go_id": protein_go_id
        }
        result_dict["data"].append(row)

    # one hour timeout
    cache.set(cache_key, result_dict, timeout=3600)

    return JsonResponse(result_dict, safe=False)


def proteome_results_report_table(request, result_id):

    if not valid_token("PROTEO1", result_id):
        return render(request, "status/404.html", status=404)

    try:
        proteome_task = ProteomeTask.objects.get(id=result_id)
    except ProteomeTask.DoesNotExist:
        return render(request, "status/404.html", status=404)

    short_id = result_id.split("-")[-1]

    if proteome_task.status in ["P", "Q"]:
        return render(
            request,
            "infer/proteome/progress.html",
            {
                "result_id": result_id,
                "short_id": short_id,
                "architecture": proteome_task.weights.architecture.name,
                "architecture_version": proteome_task.weights.architecture.version,
                "organism_sci_name": proteome_task.organism.scientific_name,
                "organism_common_name": proteome_task.organism.common_name,
                "sequence": proteome_task.sequence
            },
        )
    elif proteome_task.status == "D":

        return render(
            request,
            "infer/proteome/table.html",
            {
                "result_id": result_id,
                "short_id": short_id,
                "architecture": proteome_task.weights.architecture.name,
                "architecture_version": proteome_task.weights.architecture.version,
                "organism_sci_name": proteome_task.organism.scientific_name,
                "organism_common_name": proteome_task.organism.common_name,
                "sequence": proteome_task.sequence
            },
        )
    else:
        return render(request, "status/500.html", status=500)


def proteome_results_report_prob_hist(request, result_id):

    if not valid_token("PROTEO1", result_id):
        return render(request, "status/404.html", status=404)

    try:
        proteome_task = ProteomeTask.objects.get(id=result_id)
    except ProteomeTask.DoesNotExist:
        return render(request, "status/404.html", status=404)

    short_id = result_id.split("-")[-1]

    if proteome_task.status in ["P", "Q"]:
        return render(
            request,
            "infer/proteome/progress.html",
            {
             "result_id": result_id,
             "short_id": short_id,
             "architecture": proteome_task.weights.architecture.name,
             "architecture_version": proteome_task.weights.architecture.version,
             "organism_sci_name": proteome_task.organism.scientific_name,
             "organism_common_name": proteome_task.organism.common_name,
             "sequence": proteome_task.sequence
            },
        )
    elif proteome_task.status == "D":

        probability_histogram = get_proteome_probability_histogram(result_id)

        return render(
            request,
            "infer/proteome/prob_hist.html",
            {
             "result_id": result_id,
             "short_id": short_id,
             "probability_histogram": probability_histogram,
             "architecture": proteome_task.weights.architecture.name,
             "architecture_version": proteome_task.weights.architecture.version,
             "organism_sci_name": proteome_task.organism.scientific_name,
             "organism_common_name": proteome_task.organism.common_name,
             "sequence": proteome_task.sequence
             },
        )
    else:
        return render(request, "status/500.html", status=500)


def proteome_results_report_go_freq(request, result_id):

    if not valid_token("PROTEO1", result_id):
        return render(request, "status/404.html", status=404)

    try:
        proteome_task = ProteomeTask.objects.get(id=result_id)
    except ProteomeTask.DoesNotExist:
        return render(request, "status/404.html", status=404)

    short_id = result_id.split("-")[-1]

    if proteome_task.status in ["P", "Q"]:
        return render(
            request,
            "infer/proteome/progress.html",
            {
                "result_id": result_id,
                "short_id": short_id,
                "architecture": proteome_task.weights.architecture.name,
                "architecture_version": proteome_task.weights.architecture.version,
                "organism_sci_name": proteome_task.organism.scientific_name,
                "organism_common_name": proteome_task.organism.common_name,
                "sequence": proteome_task.sequence
            },
        )
    elif proteome_task.status == "D":

        go_frequency = json.loads(get_proteome_go_frequency(result_id))
        go_frequency_counts = json.dumps(go_frequency['counts'])
        go_frequency_labels = json.dumps(go_frequency['labels'])

        go_frequency_zip = [(label, count) for label, count in zip(go_frequency['labels'], go_frequency['counts'])]

        return render(
            request,
            "infer/proteome/go_freq.html",
            {
                "result_id": result_id,
                "short_id": short_id,
                "go_frequency_counts": go_frequency_counts,
                "go_frequency_labels": go_frequency_labels,
                "go_frequency_zip": go_frequency_zip,
                "architecture": proteome_task.weights.architecture.name,
                "architecture_version": proteome_task.weights.architecture.version,
                "organism_sci_name": proteome_task.organism.scientific_name,
                "organism_common_name": proteome_task.organism.common_name,
                "sequence": proteome_task.sequence
            },
        )
    else:
        return render(request, "status/500.html", status=500)


def post_panther_enrichment(gene_input_list, organism, annot_dataset, enrichment_test_type, correction):
    """Perform request to pantherdb API for enrichment analysis."""
    url = "http://pantherdb.org/services/oai/pantherdb/enrich/overrep"

    params = {
        "geneInputList": json.dumps(gene_input_list),
        "organism": organism,
        "annotDataSet": annot_dataset,
        "enrichmentTestType": enrichment_test_type,
        "correction": correction,
    }

    return requests.post(url, params=params)


def proteome_results_report_go_enrich(request, result_id):
    if not valid_token("PROTEO1", result_id):
        return render(request, "status/404.html", status=404)

    try:
        proteome_task = ProteomeTask.objects.get(id=result_id)
    except ProteomeTask.DoesNotExist:
        return render(request, "status/404.html", status=404)

    short_id = result_id.split("-")[-1]

    if request.method == "POST":
        form = GOEnrichmentForm(request.POST)

        if form.is_valid():

            task = ProteomeTask.objects.get(id=result_id)

            cache_key = hash_dict({"organism": task.organism.ncbi_taxon,
                         "annotation_dataset": form.cleaned_data["annotation_dataset"],
                         "enrichment_test_type": form.cleaned_data["enrichment_test_type"],
                         "correction": form.cleaned_data["correction"]})
            cache_key = f"PROTEOME_GO_ENRICH_V2_{cache_key}"
            cache_api_result = cache.get(cache_key)

            if cache_api_result:
                return render(request,
                              "infer/proteome/go_enrich_table.html",
                              context={"result_id": result_id, "short_id": short_id, "datatable": cache_api_result, "architecture": proteome_task.weights.architecture.name,
                "architecture_version": proteome_task.weights.architecture.version,
                "organism_sci_name": proteome_task.organism.scientific_name,
                "organism_common_name": proteome_task.organism.common_name,
                "sequence": proteome_task.sequence})

            else:

                results = ProteomeResult.objects.filter(task_id=result_id,
                                                        probability__gte=form.cleaned_data['threshold']/100).all()

                gene_input_list = ",".join([result.protein.id for result in results])

                api_result = post_panther_enrichment(gene_input_list, task.organism.ncbi_taxon, form.cleaned_data["annotation_dataset"],
                                        form.cleaned_data["enrichment_test_type"], form.cleaned_data["correction"])

                datatable = []

                api_result_json = api_result.json()

                for row in api_result_json['results']['result']:

                    if "id" in row["term"]:
                        if isinstance(row["term"]["id"], tuple):
                            go_id = row["term"]["id"][0]
                        else:
                            go_id = row["term"]["id"]
                    else:
                        go_id = ""

                    datatable.append({
                        "go_id": go_id,
                        "go_label": row["term"]["label"],
                        "fold_enrichment": row["fold_enrichment"],
                        "fdr": row["fdr"],
                        "expected": row["expected"],
                        "p_value": row["pValue"],
                        "plus_minus": row["plus_minus"],
                    })

                datatable = sorted(datatable, key=lambda x: x["p_value"]**2 + x["fdr"]**2)

                cache.set(cache_key, datatable)

                return render(request,
                              "infer/proteome/go_enrich_table.html",
                              context={"result_id": result_id, "short_id": short_id, "datatable": datatable, "architecture": proteome_task.weights.architecture.name,
                "architecture_version": proteome_task.weights.architecture.version,
                "organism_sci_name": proteome_task.organism.scientific_name,
                "organism_common_name": proteome_task.organism.common_name,
                "sequence": proteome_task.sequence})
        else:
            return redirect("proteome_report_go_enrich", result_id)

    else:

        form = GOEnrichmentForm()

    return render(request, "infer/proteome/go_enrich.html", {
                "result_id": result_id,
                "short_id": short_id,
                "architecture": proteome_task.weights.architecture.name,
                "architecture_version": proteome_task.weights.architecture.version,
                "organism_sci_name": proteome_task.organism.scientific_name,
                "organism_common_name": proteome_task.organism.common_name,
                "sequence": proteome_task.sequence,
                "form": form
            })


# cache for a week
#@cache_page(60 * 60 * 24 * 7)
def validate_org_arch(request, org_id, arch_id):
    try:
        weights = Weights.objects.filter(architecture_id=arch_id, enabled=True).latest(
            'pub_date')
    except Weights.DoesNotExist:
        return JsonResponse({"count": -1, "weights": None, "weights_id": "no weights"}, status=404)

    count = Protein.objects.filter(organism_id=org_id, embedding__weights_id=weights.id).count()

    return JsonResponse({"count": count, "weights": weights.name, "organism": org_id, "weights_id": weights.id}, status=200)


class Echo:
    """An object that implements just the write method of the file-like
    interface.
    """

    def write(self, value):
        """Write the value by returning it, instead of storing in a buffer."""
        return value


def proteome_exporter_csv(proteome_result, result_id):

    # This key locks streaming, so that only one download of one report happens at a time.
    cache.set(f"STREAMING_CSV_{result_id}", True)

    try:
        yield "UniProt AC", "Protein Name", "Gene Name", "Interaction Probability"
        for row in proteome_result:
            yield row.protein.id, row.protein.protein_name, row.protein.gene_name, row.probability
    finally:
        cache.set(f"STREAMING_CSV_{result_id}", False)


def proteome_export_csv(request, result_id):
    if not valid_token("PROTEO1", result_id):
        return render(request, "status/404.html", status=404)

    try:
        proteome_result = ProteomeResult.objects.filter(task__id=result_id).all()
    except ProteomeTask.DoesNotExist:
        return render(request, "status/404.html", status=404)

    if cache.get(f"STREAMING_CSV_{result_id}") and cache.get(f"STREAMING_CSV_{result_id}") is True:
        print(cache.get(f"STREAMING_CSV_{result_id}"))
        return render(request, "status/429.html", status=429)

    pseudo_buffer = Echo()
    writer = csv.writer(pseudo_buffer)

    exp_iter = proteome_exporter_csv(proteome_result, result_id)

    return StreamingHttpResponse(
        (writer.writerow(row) for row in exp_iter),
        content_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{result_id}.csv"'},
    )


def parse_pairs(pairs):
    for pair in pairs.split("\n"):
        yield pair.strip().split(",")


def diagnostics_page(request):
    if request.user.is_superuser and request.method == "POST":
        form = DiagnosticForm(request.POST)

        if form.is_valid():
            try:
               weights = Weights.objects.get(pk=form.cleaned_data['weights'])
            except Weights.DoesNotExist:
               return render(request, "status/404.html", status=404)

            model = get_model(weights)

            protein1_vecs = []
            protein2_vecs = []
            y_trues = []

            for upkb_ac1, upkb_ac2, label in parse_pairs(form.cleaned_data['pairs']):
                protein1 = ProteinEmbedding.objects.filter(protein__id=upkb_ac1, weights_id=weights.id).first()
                protein2 = ProteinEmbedding.objects.filter(protein__id=upkb_ac2, weights_id=weights.id).first()

                if None in [protein1, protein2]:
                    continue

                protein1_vec = json.loads(protein1.vector)
                protein2_vec = json.loads(protein2.vector)
                y_trues.append(int(label))

                protein1_vecs.append(protein1_vec)
                protein2_vecs.append(protein2_vec)

            protein1_vecs, protein2_vecs = torch.tensor(protein1_vecs), torch.tensor(protein2_vecs)

            y_hats = infer_batch(model, weights, protein1_vecs, protein2_vecs)

            auroc = metrics.roc_auc_score(y_trues, y_hats)

            return JsonResponse({"auroc": auroc, "y_trues": y_trues, "y_hats": y_hats})
        else:
            return render(request, "status/404.html", status=404)

    elif request.user.is_superuser and request.method == "GET":
        form = DiagnosticForm()

        return render(request, "infer/diagnostics.html", {"form": form})
    else:
        return render(request, "status/404.html", status=404)
