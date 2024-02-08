from django.core.cache import cache
from infer.tasks import rapppid, intrepppid
from infer.models import ProteomeTask, ProteomeResult, ProteomeResultStats, get_all_task_result
from infer import models

import sentencepiece as sp
import numpy as np
from torch import nn
import random
import os
import json
import torch

from typing import Iterable, List
from datetime import datetime
from collections import Counter


def seed_everything(torch_seed: int):
    # https://github.com/qhd1996/seed-everything

    random.seed(torch_seed)
    os.environ["PYTHONHASHSEED"] = str(torch_seed)
    np.random.seed(torch_seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_embeddings(weights: models.Weights, seqs: Iterable[str]):
    seed_everything(8675309)

    if weights.architecture.name == "RAPPPID":
        with torch.no_grad():
            model = rapppid.load_chkpt(weights.chkpt.path)

            spp = sp.SentencePieceProcessor(model_file=weights.spm.path)

            toks = rapppid.process_seqs(spp, seqs, model.trunc_len)

            out = model(toks)

            return out

    elif weights.architecture.name == "INTREPPPID":
        with torch.no_grad():
            model = intrepppid.load_chkpt(weights.chkpt.path, weights.args)

            spp = sp.SentencePieceProcessor(model_file=weights.spm.path)

            toks = intrepppid.process_seqs(spp, seqs, weights.trunc_len)

            out = model.encoder(toks)

            return out
    else:
        raise ValueError("Unexpected architecture.")


def infer_batch(
    model: nn.Module,
    weights: models.Weights,
    user_embedding_batch: torch.Tensor,
    batch: torch.Tensor,
) -> List:
    if weights.architecture.name == "RAPPPID":
        with torch.no_grad():
            out = model.class_head(user_embedding_batch, batch)
            out = torch.sigmoid(out).tolist()
    elif weights.architecture.name == "INTREPPPID":
        with torch.no_grad():
            y_hat_logits = model.head(user_embedding_batch, batch)
            out = torch.sigmoid(y_hat_logits.float()).flatten().detach().numpy().astype(np.float32).tolist()
    else:
        raise ValueError("Unexpected architecture.")

    return out


def get_model(weights):

    if weights.architecture.name == "RAPPPID":
        model = rapppid.load_chkpt(weights.chkpt.path)
    elif weights.architecture.name == "INTREPPPID":
        model = intrepppid.load_chkpt(weights.chkpt.path, weights.args)
    else:
        raise ValueError(f"Unexpected architecture {weights.architecture.name}.")

    return model


def pairwise_predict(
    result_id: str, weights: models.Weights, seqs: str, organism: models.Organism
) -> List[float]:
    seed_everything(8675309)

    proteome_task = ProteomeTask.objects.get(id=result_id)
    proteome_task.status = "P"
    proteome_task.save()

    batch_size = weights.args["batch_size"] if "batch_size" in weights.args else 1

    user_embedding = compute_embeddings(weights, [seqs])
    user_embedding_batch = torch.tensor(user_embedding.tolist())

    batch = []
    outs = []

    model = get_model(weights)

    idx = 1
    proteins = models.Protein.objects.filter(organism=organism)

    num_proteins = len(proteins)

    cache.set(
        f"STATUS-{result_id}",
        json.dumps(
            {"total_proteins": num_proteins, "computed_proteins": 0, "status": "P"}
        ),
        timeout=None,
    )

    for protein in proteins:

        try:
            vector = json.loads(
                protein.embedding.filter(weights_id=weights.id).first().vector
            )
            batch.append(vector)
        except Exception:
            print("MISSING VECTOR")

        if len(batch) >= batch_size:
            batch = torch.tensor(batch)
            print(f"BATCH {idx}/{num_proteins}")

            out = infer_batch(model, weights, user_embedding_batch, batch)
            if weights.architecture.name == "RAPPPID":
                out = [o[0] for o in out]  # flatten out
            else:
                out = [o for o in out]
            outs += out

            batch = []

            cache.set(
                f"STATUS-{result_id}",
                json.dumps(
                    {
                        "total_proteins": num_proteins,
                        "computed_proteins": idx,
                        "status": "P",
                    }
                ),
                timeout=None,
            )

        idx += 1

    if len(batch) > 0:
        batch = torch.tensor(batch)
        out = infer_batch(model, weights, user_embedding_batch, batch)
        if weights.architecture.name == "RAPPPID":
            out = [o[0] for o in out]  # flatten out
        else:
            out = [o for o in out]
        outs += out
        cache.set(
            f"STATUS-{result_id}",
            json.dumps(
                {
                    "total_proteins": num_proteins,
                    "computed_proteins": idx,
                    "status": "P",
                }
            ),
            timeout=None,
        )

    proteome_results = []
    probs = []

    for protein, out in zip(proteins, outs):
        proteome_result = models.ProteomeResult(
            protein=protein, probability=out, task=proteome_task
        )
        proteome_results.append(proteome_result)
        probs.append(out)

    ProteomeResult.objects.bulk_create(proteome_results)

    # Statistics
    # -- Compute probability histogram
    histogram = np.histogram(probs, bins=[0.1 * i for i in range(11)])
    histogram_json = json.dumps(histogram[0].tolist())
    proteome_result_stats = ProteomeResultStats(task=proteome_task, statistic="probability_histogram", value=histogram_json)
    proteome_result_stats.save()

    # -- Compute GO histogram
    percentile_95 = np.percentile(probs, 95)
    go_counter = Counter()

    for protein, prob in zip(proteins, probs):
        if prob >= percentile_95:
            gos = [go.id for go in protein.gene_ontologies.all() if go.id != ""]
            go_counter.update(gos)

    top_gos_10 = go_counter.most_common(10)

    top_gos_10_labels = []
    top_gos_10_counts = []

    for label, count in top_gos_10:
        top_gos_10_labels.append(label)
        top_gos_10_counts.append(count)

    top_gos_10 = json.dumps({
        "labels": top_gos_10_labels,
        "counts": top_gos_10_counts
    })

    proteome_result_stats = ProteomeResultStats(task=proteome_task, statistic="top_gos_10", value=top_gos_10)
    proteome_result_stats.save()

    # This will cache the result
    get_all_task_result(result_id)

    proteome_task.status = "D"
    proteome_task.completion_timestamp = datetime.now()
    proteome_task.save()

    cache.set(
        f"STATUS-{result_id}",
        json.dumps(
            {
                "total_proteins": num_proteins,
                "computed_proteins": idx,
                "status": "D",
            }
        ),
        timeout=None,
    )

    return outs
