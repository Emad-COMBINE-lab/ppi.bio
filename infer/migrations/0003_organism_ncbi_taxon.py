# Generated by Django 4.1.6 on 2023-02-06 03:32

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("infer", "0002_weights_organisms"),
    ]

    operations = [
        migrations.AddField(
            model_name="organism",
            name="ncbi_taxon",
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
    ]
