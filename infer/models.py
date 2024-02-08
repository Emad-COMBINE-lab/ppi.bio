from typing import List, Dict, Any
import json

from django.contrib.postgres.aggregates import ArrayAgg
from django.core.cache import cache
from django.db import models
from django.db.models import F


class Organism(models.Model):
    ncbi_taxon = models.IntegerField(unique=True)
    common_name = models.CharField(max_length=280)
    scientific_name = models.CharField(max_length=280)

    def __str__(self):
        return f"{self.common_name}"


class Architecture(models.Model):
    name = models.CharField(max_length=280)
    version = models.CharField(max_length=20)

    def __str__(self):
        return f"{self.name} v.{self.version}"


class PPIDatabase(models.Model):
    name = models.CharField(max_length=280)
    version = models.CharField(max_length=20)
    link = models.URLField()

    def __str__(self):
        return f"{self.name} v.{self.version}"


class Dataset(models.Model):
    ppi_database = models.ForeignKey(PPIDatabase, on_delete=models.CASCADE)
    organisms = models.ManyToManyField(Organism)
    c_type = models.IntegerField()
    train_proportion = models.FloatField()
    val_proportion = models.FloatField()
    test_proportion = models.FloatField()
    neg_proportion = models.FloatField()
    score_threshold = models.FloatField()
    preloaded_protein_splits_path = models.CharField(
        default="", max_length=280, blank=True
    )
    seed = models.IntegerField()
    negatives_path = models.CharField(default="", max_length=280, blank=True)

    def __str__(self):
        try:
            return f"Dataset #{self.id} [C{self.c_type} {self.ppi_database} {self.organisms.get()}]"
        except Organism.DoesNotExist:
            return f"Dataset #{self.id} [C{self.c_type} {self.ppi_database}]"


class Weights(models.Model):
    name = models.CharField(max_length=280, unique=True)
    pub_date = models.DateTimeField("date published")
    chkpt = models.FileField(upload_to="chkpts")
    spm = models.FileField(upload_to="spms")
    log = models.FileField(upload_to="logs")
    args = models.JSONField()
    architecture = models.ForeignKey(Architecture, on_delete=models.CASCADE)
    trunc_len = models.IntegerField()
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    enabled = models.BooleanField(default=False)

    def __str__(self):
        return f'{self.architecture} model "{self.name}" trained on {self.dataset}'


class GO(models.Model):
    id = models.CharField(max_length=280, unique=True, primary_key=True)

    def __str__(self):
        return f"{self.id}"


class AlternateProteinName(models.Model):
    name = models.CharField(max_length=500)

    def __str__(self):
        return f"{self.name}"


class AlternateGeneName(models.Model):
    name = models.CharField(max_length=500)

    def __str__(self):
        return f"{self.name}"


class ProteinEmbedding(models.Model):
    vector = models.CharField(max_length=3000)
    weights = models.ForeignKey(Weights, on_delete=models.CASCADE)

    def __str__(self):
        return f"Embedding #{self.id} ({self.weights})"


class Protein(models.Model):
    id = models.CharField(max_length=10, unique=True, primary_key=True)
    reviewed = models.BooleanField()
    entry_name = models.CharField(max_length=280)
    protein_name = models.CharField(max_length=500)
    alternate_protein_names = models.ManyToManyField(AlternateProteinName)
    gene_name = models.CharField(max_length=500)
    alternate_gene_names = models.ManyToManyField(AlternateGeneName)
    organism = models.ForeignKey(Organism, on_delete=models.CASCADE)
    sequence_length = models.IntegerField()
    sequence = models.CharField(max_length=3000)
    gene_ontologies = models.ManyToManyField(GO)
    embedding = models.ManyToManyField(ProteinEmbedding)

    def __str__(self):
        return f"{self.id} ({self.protein_name})"


class ProteomeTask(models.Model):
    STATUS = (
        ("D", "DONE"),
        ("P", "IN PROGRESS"),
        ("F", "FAILED"),
        ("Q", "QUEUED"),
    )

    id = models.CharField(max_length=64, unique=True, primary_key=True)
    qid = models.UUIDField()
    status = models.CharField(max_length=1, choices=STATUS)
    organism = models.ForeignKey(Organism, on_delete=models.CASCADE)
    sequence = models.CharField(max_length=3000)
    weights = models.ForeignKey(Weights, on_delete=models.CASCADE)
    submission_timestamp = models.DateTimeField("Submission Timestamp")
    completion_timestamp = models.DateTimeField(
        "Completion Timestamp", blank=True, null=True
    )


class ProteomeResult(models.Model):
    protein = models.ForeignKey(Protein, on_delete=models.CASCADE)
    probability = models.FloatField()
    task = models.ForeignKey(ProteomeTask, on_delete=models.CASCADE)


class ProteomeResultStats(models.Model):
    task = models.ForeignKey(ProteomeTask, on_delete=models.CASCADE)
    statistic = models.CharField(max_length=3000)
    value = models.CharField(max_length=3000)


def get_all_task_result(result_id: str) -> List[Dict[str, Any]]:

    cache_key = f"PROTEOME_RESULTS_{result_id}"
    cache_result = cache.get(cache_key)

    if cache_result:
        return cache_result
    else:
        results = (
            ProteomeResult.objects.filter(task_id=result_id, protein__reviewed=True)
            .exclude(protein__gene_name="")
            .exclude(protein__protein_name="MHC class I antigen")
            .order_by("-probability")
            .all()
            .annotate(
                uniprot_ac=F("protein_id"),
                gene_name=F("protein__gene_name"),
                protein_name=F("protein__protein_name"),
                protein_go_id=ArrayAgg("protein__gene_ontologies"))
            .values(
                "protein", "uniprot_ac", "probability", "gene_name", "protein_name", "protein_go_id"
            )
        )
        cache.set(cache_key, results, timeout=3600)
        return results


def get_proteome_stats(result_id: str) -> List[ProteomeResultStats]:

    cache_key = f"PROTEOME_STAT_RESULTS_{result_id}"
    cache_stats = cache.get(cache_key)

    if cache_stats:
        return cache_stats
    else:
        stats = ProteomeResultStats.objects.filter(task_id=result_id).all()
        cache.set(cache_key, stats, timeout=3600)
        return stats


def get_proteome_probability_histogram(result_id: str) -> str:
    proteome_stats = get_proteome_stats(result_id)

    probility_histogram = json.dumps([0 for _ in range(10)])

    for proteome_stat in proteome_stats:
        if proteome_stat.statistic == "probability_histogram":
            probility_histogram = proteome_stat.value
            break

    return probility_histogram


def get_proteome_go_frequency(result_id: str) -> str:
    proteome_stats = get_proteome_stats(result_id)

    go_frequency = json.dumps({
        'labels': ["?" for _ in range(10)],
        'counts': [0 for _ in range(10)]
    })

    for proteome_stat in proteome_stats:
        if proteome_stat.statistic == "top_gos_10":
            go_frequency = proteome_stat.value
            break

    return go_frequency
