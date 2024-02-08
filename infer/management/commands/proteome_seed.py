from django.core.management.base import BaseCommand, CommandError
from django_q.tasks import async_task
from infer.models import AlternateProteinName, AlternateGeneName, GO, Organism, Protein
from requests.adapters import HTTPAdapter, Retry
import requests
import re


def download_uniprot(taxon: int):
    re_next_link = re.compile(r'<(.+)>; rel="next"')
    retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retries))

    def get_next_link(headers):
        if "Link" in headers:
            match = re_next_link.match(headers["Link"])
            if match:
                return match.group(1)

    def get_batch(batch_url):
        while batch_url:
            response = session.get(batch_url)
            response.raise_for_status()
            total = response.headers["x-total-results"]
            yield response, total
            batch_url = get_next_link(response.headers)

    download_path = f"https://rest.uniprot.org/uniprotkb/search?compressed=false&fields=accession%2Creviewed%2Cid%2Cprotein_name%2Cgene_names%2Corganism_name%2Clength%2Csequence%2Cgo_id&format=tsv&query=%28%2A%29%20AND%20%28model_organism%3A{taxon}%29&size=500"

    for batch, total in get_batch(download_path):
        for line in batch.text.splitlines()[1:]:
            (
                entry,
                reviewed,
                entry_name,
                protein_names,
                gene_names,
                organism,
                length,
                sequence,
                go_ids,
            ) = line.split("\t")

            entry_name = entry_name[:280]

            gos = []

            for go_id in go_ids.split("; "):
                go, _ = GO.objects.get_or_create(id=go_id)
                gos.append(go)

            protein_names = [
                _.replace("(", "").replace(")", "") for _ in protein_names.split(" (")
            ]
            alternate_protein_names = []

            for protein_name in protein_names[1:]:
                alt_name, _ = AlternateProteinName.objects.get_or_create(
                    name=protein_name[:500]
                )
                alternate_protein_names.append(alt_name)

            gene_names = [
                _.replace("(", "").replace(")", "") for _ in gene_names.split(" ")
            ]
            alternate_gene_names = []

            for gene_name in gene_names[1:]:
                alt_name, _ = AlternateGeneName.objects.get_or_create(
                    name=gene_name[:500]
                )
                alternate_gene_names.append(alt_name)

            organism = Organism.objects.get(ncbi_taxon=taxon)

            protein, _ = Protein.objects.update_or_create(
                id=entry,
                reviewed=bool(reviewed),
                entry_name=entry_name,
                protein_name=protein_names[0][:500],
                gene_name=gene_names[0][:500],
                sequence_length=int(length),
                sequence=sequence[:3000],
                organism=organism,
            )

            for alternate_protein_name in alternate_protein_names:
                protein.alternate_protein_names.add(alternate_protein_name)

            for alternate_gene_name in alternate_gene_names:
                protein.alternate_gene_names.add(alternate_gene_name)

            for go in gos:
                protein.gene_ontologies.add(go)

            protein.save()


class Command(BaseCommand):
    help = "Downloads protein sequences from UniProt for one organism."

    def add_arguments(self, parser):
        parser.add_argument("taxon_id", nargs="+", type=int)

    def proteome_seed(self, taxon: int):
        try:
            Organism.objects.get(ncbi_taxon=taxon)
        except Organism.DoesNotExist:
            return self.stdout.write(
                self.style.ERROR(f"Organism with Taxon ID {taxon} does not exist.")
            )

        try:
            taxon = int(taxon)
            task_id = async_task(download_uniprot, taxon)
            return self.stdout.write(
                self.style.SUCCESS(
                    f"Database is being seeded w/ proteins from taxon ID {taxon}! "
                    f"[Task ID: {task_id}]"
                )
            )
        except Exception:
            return self.stdout.write(self.style.ERROR(f"Invalid Taxon ID: {taxon}"))

    def handle(self, *args, **options):
        for taxon_id in options["taxon_id"]:
            self.proteome_seed(taxon_id)
