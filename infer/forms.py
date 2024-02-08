import django.db
from django import forms
from infer.models import Organism, Architecture, Weights


class ProteomeSubmitForm(forms.Form):

    try:
        organism_choices = [(organism.id, f"{organism.scientific_name} ({organism.common_name})") for organism in
                            Organism.objects.all()]
    except django.db.ProgrammingError:
        organism_choices = [('', '')]

    organism = forms.ChoiceField(choices=organism_choices, required=True, label="Organism", help_text="Interactions between the proteome of the selected organism and the amino acid sequence you specify below will be computed.")

    try:
        architecture_choices = [(architecture.id, f"{architecture.name} V{architecture.version}") for architecture in
                            Architecture.objects.all()]
    except django.db.ProgrammingError:
        architecture_choices = [('', '')]

    architecture = forms.ChoiceField(choices=architecture_choices, required=True, label="Inference Model", help_text="Which PPI inference model to use during the analysis. INTREPPPID is the newest model and recommend.")

    seq = forms.CharField(
        label="Amino Acid Sequence", max_length=1500, widget=forms.Textarea, required=True,
        help_text="PPIs between the protein or peptide encoded by this amino acid sequence and the proteome of the organism specified above are computed."
    )


class GOEnrichmentForm(forms.Form):

    threshold = forms.IntegerField(min_value=1, max_value=100, required=True, step_size=1, initial=90)

    annotation_dataset_choices = [
        ("GO:0003674", "Molecular Function"),
        ("GO:0008150", "Biological Process"),
        ("GO:0005575", "Cellular Component")
    ]

    annotation_dataset = forms.ChoiceField(choices=annotation_dataset_choices, required=True)

    enrichment_test_type_choices = [
        ("FISHER", "Fisher's Exact"),
        ("BINOMIAL", "Binomial distribution test")
    ]
    enrichment_test_type = forms.ChoiceField(choices=enrichment_test_type_choices, required=True, initial="FISHER")

    correction_choices = [
        ("FDR", "Benjamini-Hochberg FDR correction"),
        ("BONFERRONI", "Bonferroni correction"),
        ("NONE", "No correction")
    ]
    correction = forms.ChoiceField(choices=correction_choices, required=True, initial="FISHER")


class DiagnosticForm(forms.Form):

    try:
        weights_choices = [(weights.id, f"{weights.name} ({weights.architecture.name} V{weights.architecture.version})") for weights in
                            Weights.objects.all()]
    except django.db.ProgrammingError:
        weights_choices = [('', '')]

    weights = forms.ChoiceField(choices=[(1, "1")], required=True)

    pairs = forms.CharField(
        label="Test Pairs", max_length=160_000, widget=forms.Textarea, required=True,
        help_text="UPKB AC1,UPKB AC2,Label"
    )