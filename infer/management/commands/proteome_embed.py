from django.core.management.base import BaseCommand, CommandError
from django_q.tasks import async_task
from infer.models import Protein, Weights, Organism, ProteinEmbedding
from infer.tasks.infer import compute_embeddings
import json


def attach_proteins(weights, taxon, batch_min, batch_max):
    batch = Protein.objects.order_by("id").filter(organism__ncbi_taxon=taxon)[
        batch_min:batch_max
    ]
    seq_len = weights.trunc_len

    if len(batch) == 0:
        print("Can't attach protein embeddings, batch is size zero")

    vecs = (
        compute_embeddings(weights, [_.sequence[:seq_len] for _ in batch])
        .numpy()
        .tolist()
    )

    for protein, vec in zip(batch, vecs):
        embedding, _ = ProteinEmbedding.objects.get_or_create(
            vector=json.dumps(vec), weights=weights
        )
        protein.embedding.add(embedding)
        protein.save()


class Command(BaseCommand):
    help = "Compute protein embeddings of the protein sequences for one organism."

    def add_arguments(self, parser):
        parser.add_argument("taxon_id", nargs="+", type=int)
        parser.add_argument("weight_name", type=str)

    def proteome_embed(self, taxon: int, weight_name: str):
        try:
            organism = Organism.objects.get(ncbi_taxon=taxon)
        except Organism.DoesNotExist:
            return self.stdout.write(
                self.style.ERROR(f"Organism with Taxon ID {taxon} does not exist.")
            )

        weights = Weights.objects.get(name=weight_name)

        batch_size = weights.args["batch_size"] if "batch_size" in weights.args else 1

        number_proteins = Protein.objects.filter(organism=organism).count()

        for i in range((number_proteins // batch_size) + 1):
            async_task(
                attach_proteins,
                weights,
                taxon,
                i * batch_size,
                i * batch_size + batch_size,
            )

        print(batch_size, Protein.objects.count())

        return self.stdout.write(
            self.style.SUCCESS(
                f"Computed embeddings for proteins from taxon ID {taxon}!"
            )
        )

    def handle(self, *args, **options):
        for taxon_id in options["taxon_id"]:
            self.proteome_embed(taxon_id, options["weight_name"])
