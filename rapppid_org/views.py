from django.shortcuts import render
from django.views.decorators.cache import cache_page
from rapppid_org.settings import TEMPLATES, STATIC_URL
from glob import glob
import markdown
import os.path
import re

def index(request):
    return render(request, "index.html")


@cache_page(60 * 60)
def privacy_policy(request):
    return render(request, "privacy_policy.html")


@cache_page(60 * 60)
def cookie_policy(request):
    return render(request, "cookie_policy.html")


@cache_page(60 * 60)
def early_preview(request):
    return render(request, "early_preview.html")


@cache_page(60 * 60)
def faq(request):
    return render(request, "faq.html")


def get_help_metadata():
    templates_dir = TEMPLATES[0]["DIRS"][0]
    help_dir = str(templates_dir / "help/*.md")
    markdown_files = glob(help_dir)

    metas = []

    for markdown_file in markdown_files:
        with open(markdown_file) as f:
            data = f.read()
            md = markdown.Markdown(extensions=['meta'])
            _ = md.convert(data)
            file_path = ".".join(os.path.basename(os.path.normpath(markdown_file)).split('.')[:-1])
            meta = md.Meta
            meta["file_path"] = file_path
            metas.append(meta)

    return metas


def get_help_page(help_file):
    templates_dir = TEMPLATES[0]["DIRS"][0]
    static_url = STATIC_URL

    # important to make help_file alphanumeric to help mitigate the risk of
    # printing arbitrary files
    help_file = re.sub('[^a-zA-Z0-9\_\-]', '', help_file)

    help_path = str(templates_dir / f"help/{help_file}.md")

    try:
        with open(help_path) as f:
            data = f.read()

            # replace { STATIC_URL } with the static url
            data = data.replace("{STATIC_URL}", STATIC_URL)

            md = markdown.Markdown(extensions=['meta'])
            help_body = md.convert(data)
            help_meta = md.Meta
    except FileNotFoundError:
        return None, None

    return help_meta, help_body


#@cache_page(60 * 60 * 24 * 7)
def help_index(request, help_file = None):

    if help_file is None:
        metas = get_help_metadata()
        print(metas)
        return render(request, "help/index.html", context={"metas": metas})
    else:
        help_meta, help_body = get_help_page(help_file)

        if help_meta is None and help_body is None:
            return render(request, "status/404.html", status=404)

        return render(request,
                      "help/page.html",
                      context={"help_meta": help_meta, "help_body": help_body})
