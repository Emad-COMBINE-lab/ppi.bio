# Generated by Django 4.1.6 on 2024-01-04 20:44

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("infer", "0021_alter_proteometask_id"),
    ]

    operations = [
        migrations.AddField(
            model_name="weights",
            name="enabled",
            field=models.BooleanField(default=False),
        ),
    ]