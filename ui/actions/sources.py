from ui.actions.export_state import (
    csv,
    html,
    json,
    re,
    time,
    datetime,
    BytesIO,
    StringIO,
    Path,
    st,
    components,
    Document,
    WD_ALIGN_PARAGRAPH,
    DocxInches,
    DocxPt,
    Workbook,
    Alignment,
    Border,
    Font,
    PatternFill,
    Side,
    Presentation,
    PptxInches,
    PptxPt,
    colors,
    letter,
    ParagraphStyle,
    getSampleStyleSheet,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    clean_preview_text,
    deduplicate_sources,
    make_static_file_link,
    normalize_source_item,
    SIDEBAR_SOURCES_KEY,
    SOURCE_PREVIEW_LIMIT,
    EXPORT_TIMESTAMP_FORMAT,
    PENDING_EXPORT_REQUEST_KEY,
    READY_EXPORT_FILE_KEY,
    EXPORT_FRONTEND_FLUSH_SECONDS,
    ACTION_DEBUG_FILE,
    append_action_debug,
    EXPORT_TYPES,
    safe_html,
    copy_button,
    get_export_timestamp,
    title_case_words,
    make_response_title,
    make_filename_slug,
    make_export_filename,
    get_question_for_message,
    normalize_answer_text,
    is_bullet_line,
    is_numbered_line,
    strip_list_marker,
    split_answer_lines,
    get_source_entries,
    format_source_line,
    build_export_text,
    export_txt,
    export_md,
    export_csv,
    export_xlsx,
    export_xls,
    add_docx_paragraph,
    export_docx,
    get_pdf_styles,
    add_pdf_answer_lines,
    export_pdf,
    split_text_for_slides,
    export_pptx,
    export_ppt,
    get_export_file,
    toggle_export_panel,
    clear_export_runtime_state,
    queue_export_request,
    get_pending_export_request,
    get_ready_export_file,
    display_export_loading,
    prepare_pending_export,
    display_ready_export,
    display_export_buttons,
)



def prepare_sources_for_drawer(sources):
    # Clean and deduplicate source files.
    return deduplicate_sources(sources or [], max_sources=None)


def get_optional_source_url(source):
    # Use web/static URL when source metadata already has one.
    if not isinstance(source, dict):
        return ""

    metadata = source.get("metadata") or {}

    for key in ["url", "link", "file_url", "source_url", "path_url"]:
        value = source.get(key) or metadata.get(key)

        if value and str(value).strip().startswith(("http://", "https://", "/")):
            return str(value).strip()

    return ""


def build_source_card_html(index, source):
    # Build one source drawer card.
    source = normalize_source_item(source)

    if not source:
        return ""

    name = source.get("source", "Unknown source")
    page = source.get("page", "N/A")
    preview = clean_preview_text(source.get("preview"), limit=SOURCE_PREVIEW_LIMIT)
    link = make_static_file_link(source.get("source_path"), page=page) or get_optional_source_url(source)

    safe_name = safe_html(name)
    safe_page = safe_html(page)
    safe_preview = safe_html(preview)

    if link:
        safe_link = safe_html(link)
        title_html = f'<a class="right-source-name" href="{safe_link}" target="_blank" rel="noopener noreferrer">{safe_name}</a>'
        preview_html = f'<a class="right-source-preview" href="{safe_link}" target="_blank" rel="noopener noreferrer">{safe_preview}</a>'
    else:
        title_html = f'<span class="right-source-name">{safe_name}</span>'
        preview_html = f'<div class="right-source-preview right-source-preview-disabled">{safe_preview}</div>'

    return "\n".join([
        '<div class="right-source-card">',
        '<div class="right-source-card-header">',
        f'<span class="right-source-number">{index}</span>',
        title_html,
        '</div>',
        f'<div class="right-source-page">Page: {safe_page}</div>',
        preview_html,
        '</div>',
    ])


def render_source_cards_html(sources):
    # Render drawer source card HTML.
    cards = []

    for index, source in enumerate(prepare_sources_for_drawer(sources), start=1):
        card_html = build_source_card_html(index, source)

        if card_html:
            cards.append(card_html)

    if not cards:
        return '<div class="right-source-empty">No sources available for this answer.</div>'

    return "\n".join(cards)


def show_sources_in_sidebar(sources):
    # Store current answer sources for the sidebar Sources block.
    st.session_state[SIDEBAR_SOURCES_KEY] = prepare_sources_for_drawer(sources)


def display_sources_button(sources, unique_key):
    # Send current answer sources to the left sidebar instead of opening a drawer.
    st.button(
        "▤",
        key=f"sources_{unique_key}",
        help="Show sources in sidebar",
        disabled=not bool(sources),
        on_click=show_sources_in_sidebar,
        args=(sources,),
    )


# Public names exported by this compatibility/refactor module.
__all__ = [
    'csv',
    'html',
    'json',
    're',
    'time',
    'datetime',
    'BytesIO',
    'StringIO',
    'Path',
    'st',
    'components',
    'Document',
    'WD_ALIGN_PARAGRAPH',
    'DocxInches',
    'DocxPt',
    'Workbook',
    'Alignment',
    'Border',
    'Font',
    'PatternFill',
    'Side',
    'Presentation',
    'PptxInches',
    'PptxPt',
    'colors',
    'letter',
    'ParagraphStyle',
    'getSampleStyleSheet',
    'Paragraph',
    'SimpleDocTemplate',
    'Spacer',
    'clean_preview_text',
    'deduplicate_sources',
    'make_static_file_link',
    'normalize_source_item',
    'SIDEBAR_SOURCES_KEY',
    'SOURCE_PREVIEW_LIMIT',
    'EXPORT_TIMESTAMP_FORMAT',
    'PENDING_EXPORT_REQUEST_KEY',
    'READY_EXPORT_FILE_KEY',
    'EXPORT_FRONTEND_FLUSH_SECONDS',
    'ACTION_DEBUG_FILE',
    'append_action_debug',
    'EXPORT_TYPES',
    'safe_html',
    'copy_button',
    'get_export_timestamp',
    'title_case_words',
    'make_response_title',
    'make_filename_slug',
    'make_export_filename',
    'get_question_for_message',
    'normalize_answer_text',
    'is_bullet_line',
    'is_numbered_line',
    'strip_list_marker',
    'split_answer_lines',
    'get_source_entries',
    'format_source_line',
    'build_export_text',
    'export_txt',
    'export_md',
    'export_csv',
    'export_xlsx',
    'export_xls',
    'add_docx_paragraph',
    'export_docx',
    'get_pdf_styles',
    'add_pdf_answer_lines',
    'export_pdf',
    'split_text_for_slides',
    'export_pptx',
    'export_ppt',
    'get_export_file',
    'toggle_export_panel',
    'clear_export_runtime_state',
    'queue_export_request',
    'get_pending_export_request',
    'get_ready_export_file',
    'display_export_loading',
    'prepare_pending_export',
    'display_ready_export',
    'display_export_buttons',
    'prepare_sources_for_drawer',
    'get_optional_source_url',
    'build_source_card_html',
    'render_source_cards_html',
    'show_sources_in_sidebar',
    'display_sources_button',
]
