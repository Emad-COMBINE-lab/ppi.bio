# Generated by Django 4.1.6 on 2023-12-18 01:13

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("infer", "0020_proteomeresultstats"),
    ]

    operations = [
        migrations.AlterField(
            model_name="proteometask",
            name="id",
            field=models.CharField(
                max_length=64, primary_key=True, serialize=False, unique=True
            ),
        ),
    ]
