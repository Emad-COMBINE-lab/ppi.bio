from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from django.shortcuts import redirect

from . import views


def redirect_result(request, result_id):
    return redirect("proteome_report", result_id=result_id)


urlpatterns = [
    path("", views.infer_index, name="infer_index"),
    path("proteome/submit", views.proteome_submit, name="proteome_submit"),
    path(
        "proteome/task/<str:result_id>.json",
        views.proteome_task_json,
        name="proteome_task_json",
    ),
    path(
        "proteome/results/<str:result_id>.json",
        views.proteome_results_json,
        name="proteome_results_json",
    ),
    path(
        "proteome/report/<str:result_id>",
        redirect_result
    ),
    path(
        "proteome/report/<str:result_id>/table",
        views.proteome_results_report_table,
        name="proteome_report",
    ),
    path(
        "proteome/report/<str:result_id>/prob_hist",
        views.proteome_results_report_prob_hist,
        name="proteome_report_prob_hist",
    ),
    path(
        "proteome/report/<str:result_id>/go_freq",
        views.proteome_results_report_go_freq,
        name="proteome_report_go_freq",
    ),
    path(
        "proteome/report/<str:result_id>/go_enrich",
        views.proteome_results_report_go_enrich,
        name="proteome_report_go_enrich",
    ),
    path(
        "validate_org_arch/<int:org_id>/<int:arch_id>/",
        views.validate_org_arch,
        name="validate_org_arch",
    ),
    path(
        "proteome/report/<str:result_id>/csv",
        views.proteome_export_csv
    ),
    path(
        "diagnostic",
        views.diagnostics_page
    )
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
