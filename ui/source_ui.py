"""Source normalization, deduplication, and static file links for the UI."""

import hashlib
import html
import re
import shutil
from pathlib import Path
from urllib.parse import quote

import streamlit as st

try:
    from config.settings import STATIC_DIR
except Exception:
    STATIC_DIR = "static"

STATIC_DOC_DIR = Path(STATIC_DIR) / "docs"
STATIC_URL_PREFIX = "/app/static/docs"
NO_SOURCE_TEXT = "Unknown source"
NO_PREVIEW_TEXT = "No preview available."
DEFAULT_PREVIEW_LIMIT = 250
MAX_VISIBLE_SOURCES = 5

TEXT_VIEWER_EXTENSIONS = {
    ".txt", ".md", ".csv", ".json", ".xml", ".html", ".htm", ".py", ".log"
}
PREVIEW_KEYS = ["preview", "page_content", "content", "chunk", "text", "raw_text", "document"]
NAME_KEYS = ["title", "source", "file_name", "filename", "name", "file"]
PATH_KEYS = ["source_path", "file_path", "path", "filepath", "source"]
HTML_TAG_RE = re.compile(r"<[^>]+>", flags=re.I | re.S)


def unescape_html_text(text):
    # Decode repeated HTML escaping from old saved source values.
    text = str(text or "")

    for _ in range(5):
        decoded = html.unescape(text)

        if decoded == text:
            break

        text = decoded

    return text


def normalize_whitespace(text):
    # Collapse repeated whitespace.
    return " ".join(str(text or "").split())


def strip_html_to_text(text):
    # Convert old rendered HTML source cards into readable plain text.
    text = unescape_html_text(text)
    text = text.replace("```html", " ").replace("```", " ").replace("`", " ")
    text = re.sub(r'\b(href|target|rel|title|class)\s*=\s*["\'][^"\']*["\']', " ", text, flags=re.I)
    text = HTML_TAG_RE.sub(" ", text)

    noise_words = [
        "sidebar-source-card", "sidebar-source-header", "sidebar-source-number",
        "sidebar-source-name", "sidebar-source-page", "sidebar-source-preview",
        "right-source-card", "right-source-preview", "Open source file",
        "No source file link available", "Retrieved Sources",
    ]

    for word in noise_words:
        text = text.replace(word, " ")

    text = re.sub(r"Page:\s*(N/A|\d+)", " ", text, flags=re.I)
    text = text.replace("#", " ").replace("**", " ").replace("__", " ")

    return normalize_whitespace(text)


def get_nested_value(source, key):
    # Read value from source dict or nested metadata dict.
    if not isinstance(source, dict):
        return ""

    value = source.get(key)

    if value not in [None, ""]:
        return value

    metadata = source.get("metadata")

    if isinstance(metadata, dict):
        return metadata.get(key, "")

    return ""


def normalize_page(page):
    # Safe page display value.
    page = str(page or "").strip()

    if not page or page.lower() in {"none", "nan", "null"}:
        return "N/A"

    return page


def clean_preview_text(text, limit=DEFAULT_PREVIEW_LIMIT):
    # Clean source preview text.
    cleaned = strip_html_to_text(text)

    if not cleaned:
        return NO_PREVIEW_TEXT

    if limit is not None and len(cleaned) > limit:
        return cleaned[:limit].rstrip() + "..."

    return cleaned


def clean_source_name(text, fallback=NO_SOURCE_TEXT):
    # Clean source title/file name.
    cleaned = strip_html_to_text(text)

    if not cleaned or cleaned == NO_PREVIEW_TEXT:
        return fallback

    return cleaned


def clean_path_text(value):
    # Clean local source path without treating HTML as path.
    value = strip_html_to_text(value)

    if not value or value == NO_PREVIEW_TEXT:
        return ""

    return value.strip()


def is_existing_file_path(value):
    # True kung local existing file path.
    path = clean_path_text(value)

    if not path:
        return False

    try:
        return Path(path).exists() and Path(path).is_file()
    except OSError:
        return False


def get_source_path(source):
    # Best existing file path from source metadata.
    for key in PATH_KEYS:
        value = get_nested_value(source, key)

        if is_existing_file_path(value):
            return clean_path_text(value)

    return ""


def get_source_preview(source):
    # Best preview field from source metadata.
    for key in PREVIEW_KEYS:
        value = get_nested_value(source, key)

        if value:
            return value

    return ""


def get_source_name(source):
    # Best display source name.
    source_path = get_source_path(source)

    if source_path:
        return Path(source_path).stem

    for key in NAME_KEYS:
        value = get_nested_value(source, key)

        if value:
            cleaned = clean_source_name(value)

            if cleaned != NO_SOURCE_TEXT:
                return Path(cleaned).stem if "." in Path(cleaned).name else cleaned

    return NO_SOURCE_TEXT


def normalize_source_item(source):
    # Normalize one source dict for state/UI/export.
    if not isinstance(source, dict):
        return None

    source_path = get_source_path(source)
    source_name = get_source_name(source)
    page = normalize_page(get_nested_value(source, "page"))
    preview = clean_preview_text(get_source_preview(source), limit=None)

    if preview == NO_PREVIEW_TEXT:
        preview = "No clean preview available."

    metadata = dict(source.get("metadata") or {})

    return {
        "source": source_name,
        "title": source_name,
        "page": page,
        "preview": preview,
        "page_content": preview,
        "content": preview,
        "source_path": source_path,
        "file_path": source_path,
        "path": source_path,
        "metadata": metadata,
    }


def normalize_sources_for_state(sources):
    # Normalize source list before saving to session/DB.
    normalized = []

    for source in sources or []:
        item = normalize_source_item(source)

        if item:
            normalized.append(item)

    return normalized


def get_source_identity(source):
    # Identity used to show one card per source file.
    if not isinstance(source, dict):
        return ""

    source_path = source.get("source_path") or get_source_path(source)

    if source_path:
        return str(Path(source_path.replace("\\", "/"))).lower().strip()

    source_name = source.get("source") or get_source_name(source)
    return str(source_name).lower().strip()


def deduplicate_sources(sources, max_sources=MAX_VISIBLE_SOURCES):
    # Keep first item per source file/name.
    unique = []
    seen = set()

    for source in normalize_sources_for_state(sources):
        identity = get_source_identity(source)

        if not identity or identity in seen:
            continue

        seen.add(identity)
        unique.append(source)

        if max_sources is not None and len(unique) >= max_sources:
            break

    return unique


def safe_filename(filename):
    # Safe name for static file copy.
    filename = Path(str(filename or "source_file")).name
    return re.sub(r"[^a-zA-Z0-9._-]", "_", filename) or "source_file"


def resolve_existing_source_path(source_path):
    # Resolve local file path only when it exists.
    source_path = clean_path_text(source_path)

    for candidate in [Path(source_path), Path(source_path.replace("\\", "/"))]:
        try:
            if candidate.exists() and candidate.is_file():
                return candidate
        except OSError:
            continue

    return None


def get_pdf_page_fragment(page):
    # PDF browser page fragment.
    page = normalize_page(page)

    if not page.isdigit():
        return ""

    page_number = max(int(page), 1)
    return f"#page={page_number}"


def create_text_viewer_file(source_file, destination):
    # Create HTML viewer for text source files.
    try:
        content = source_file.read_text(encoding="utf-8", errors="replace")
    except Exception:
        content = "Unable to read source file."

    safe_title = html.escape(source_file.name)
    safe_content = html.escape(content)

    viewer_html = "\n".join([
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{safe_title}</title>",
        '<link rel="stylesheet" href="source_viewer.css">',
        "</head>",
        '<body class="source-viewer-body">',
        '<main class="source-viewer-page">',
        f'<h1 class="source-viewer-title">{safe_title}</h1>',
        f'<pre class="source-viewer-content">{safe_content}</pre>',
        "</main>",
        "</body>",
        "</html>",
    ])

    destination.write_text(viewer_html, encoding="utf-8")


def ensure_static_css():
    # Copy main CSS for standalone source viewers only once per app run/folder.
    STATIC_DOC_DIR.mkdir(parents=True, exist_ok=True)
    destination = STATIC_DOC_DIR / "source_viewer.css"

    if destination.exists():
        return

    for source_css in [Path("ui/styles/main.css"), Path("main.css")]:
        if source_css.exists():
            shutil.copy2(source_css, destination)
            return

    destination.write_text("body{font-family:system-ui;background:#0f1117;color:#fff;padding:24px;}", encoding="utf-8")


def make_static_file_link(source_path, page="N/A"):
    # Copy local source file into static/docs and return browser-openable link.
    source_file = resolve_existing_source_path(source_path)

    if not source_file:
        return None

    STATIC_DOC_DIR.mkdir(parents=True, exist_ok=True)
    file_hash = hashlib.md5(str(source_file.resolve()).encode("utf-8")).hexdigest()[:10]
    safe_name = safe_filename(source_file.name)
    suffix = source_file.suffix.lower()

    if suffix in TEXT_VIEWER_EXTENSIONS:
        ensure_static_css()
        static_name = f"{file_hash}_{safe_name}.html"
        destination = STATIC_DOC_DIR / static_name

        if not destination.exists():
            create_text_viewer_file(source_file, destination)

        return f"{STATIC_URL_PREFIX}/{quote(static_name)}"

    static_name = f"{file_hash}_{safe_name}"
    destination = STATIC_DOC_DIR / static_name

    if not destination.exists():
        shutil.copy2(source_file, destination)

    link = f"{STATIC_URL_PREFIX}/{quote(static_name)}"

    if suffix == ".pdf":
        link += get_pdf_page_fragment(page)

    return link


def render_source_card(index, source, preview_limit=DEFAULT_PREVIEW_LIMIT):
    # Optional direct source card rendering for compatibility.
    item = normalize_source_item(source)

    if not item:
        return

    link = make_static_file_link(item["source_path"], page=item["page"])
    safe_name = html.escape(item["source"])
    safe_page = html.escape(item["page"])
    safe_preview = html.escape(clean_preview_text(item["preview"], limit=preview_limit))

    if link:
        safe_link = html.escape(link, quote=True)
        name_html = f'<a class="source-card-title" href="{safe_link}" target="_blank" rel="noopener noreferrer">{safe_name}</a>'
    else:
        name_html = f'<span class="source-card-title">{safe_name}</span>'

    st.markdown(
        "\n".join([
            '<div class="source-card">',
            '<div class="source-card-header">',
            f'<span class="source-card-number">{index}</span>',
            name_html,
            '</div>',
            f'<div class="source-card-page">Page: {safe_page}</div>',
            f'<div class="source-card-preview">{safe_preview}</div>',
            '</div>',
        ]),
        unsafe_allow_html=True,
    )


def display_sources(sources, preview_limit=DEFAULT_PREVIEW_LIMIT, use_popover=True):
    # Backward-compatible source display.
    sources = deduplicate_sources(sources)

    if not sources:
        return

    if use_popover:
        with st.popover("▤"):
            for index, source in enumerate(sources, start=1):
                render_source_card(index, source, preview_limit=preview_limit)
        return

    for index, source in enumerate(sources, start=1):
        render_source_card(index, source, preview_limit=preview_limit)



def get_inline_source_tooltip(source):
    # Compact hover text for browser-native preview.
    title = source.get("source") or source.get("title") or NO_SOURCE_TEXT
    page = normalize_page(source.get("page"))
    preview = clean_preview_text(source.get("preview"), limit=220)
    parts = [str(title)]

    if page != "N/A":
        parts.append(f"Page: {page}")

    if preview and preview != NO_PREVIEW_TEXT:
        parts.append(preview)

    return "\n".join(parts)


def get_inline_source_label(source, limit=34):
    # Show the source filename instead of a numeric marker.
    item = normalize_source_item(source)

    if not item:
        return NO_SOURCE_TEXT

    label = clean_source_name(item.get("source") or item.get("title"), fallback=NO_SOURCE_TEXT)
    label = Path(label).stem if Path(label).suffix else label
    label = normalize_whitespace(label) or NO_SOURCE_TEXT

    if len(label) > limit:
        return label[:limit].rstrip() + "..."

    return label


def build_inline_source_chips(sources, max_sources=5):
    # Small numbered source chips for the end of each assistant answer.
    # deduplicate_sources() prevents repeated file/source chips.
    all_sources = deduplicate_sources(sources or [], max_sources=None)
    visible_sources = all_sources[:max_sources]

    if not visible_sources:
        return ""

    chips = []

    for index, source in enumerate(visible_sources, start=1):
        title = html.escape(get_inline_source_tooltip(source), quote=True)
        label = html.escape(get_inline_source_label(source), quote=True)
        link = make_static_file_link(source.get("source_path"), page=source.get("page"))

        if link:
            safe_link = html.escape(link, quote=True)
            chips.append(
                f'<a class="citation-chip" href="{safe_link}" title="{title}" '
                f'target="_blank" rel="noopener noreferrer">{label}</a>'
            )
        else:
            chips.append(f'<span class="citation-chip" title="{title}">{label}</span>')

    remaining = len(all_sources) - len(visible_sources)

    if remaining > 0:
        chips.append(f'<span class="citation-chip citation-more" title="Additional sources">+{remaining}</span>')

    return '<span class="citation-chip-row">' + "".join(chips) + '</span>'
