import logging

from django.contrib.admin import *
from django.contrib.admin import AdminSite

from infer.models import (
    Organism,
    Architecture,
    Weights,
    Dataset,
    PPIDatabase,
    Protein,
    AlternateProteinName,
    AlternateGeneName,
    GO,
)


logger = logging.getLogger(__name__)


class RapppidAdminSite(AdminSite):
    site_header = "PPI.bio Admin"

    def __init__(self, *args, **kwargs):
        super(RapppidAdminSite, self).__init__(*args, **kwargs)
        self._registry.update(site._registry)  # PART 2

    def get_urls(self):
        urls = super().get_urls()
        my_urls = []
        return my_urls + urls


admin_site = RapppidAdminSite(name="RapppidAdminSite")


class OrganismAdmin(ModelAdmin):
    pass


class ArchitectureAdmin(ModelAdmin):
    pass


class WeightsAdmin(ModelAdmin):
    pass


class DatasetAdmin(ModelAdmin):
    pass


class PPIDatabaseAdmin(ModelAdmin):
    pass


class ProteinAdmin(ModelAdmin):
    pass


class GOAdmin(ModelAdmin):
    pass


class AlternateProteinNameAdmin(ModelAdmin):
    pass


class AlternateGeneNameAdmin(ModelAdmin):
    pass


admin_site.register(Organism, OrganismAdmin)
admin_site.register(Architecture, ArchitectureAdmin)
admin_site.register(Weights, WeightsAdmin)
admin_site.register(Dataset, DatasetAdmin)
admin_site.register(PPIDatabase, PPIDatabaseAdmin)
admin_site.register(Protein, ProteinAdmin)
admin_site.register(GO, GOAdmin)
admin_site.register(AlternateProteinName, AlternateProteinNameAdmin)
admin_site.register(AlternateGeneName, AlternateGeneNameAdmin)
