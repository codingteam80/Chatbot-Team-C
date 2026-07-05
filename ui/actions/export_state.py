from ui.actions.export_files import (
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
)




def toggle_export_panel(unique_key):
    # Open/close the inline export panel for one answer.
    current_key = st.session_state.get("open_export_panel_key")

    if str(current_key) == str(unique_key):
        st.session_state.pop("open_export_panel_key", None)
        clear_export_runtime_state()
        return

    clear_export_runtime_state()
    st.session_state.open_export_panel_key = str(unique_key)


def clear_export_runtime_state():
    # Clear export request/result state when switching panels.
    st.session_state.pop(PENDING_EXPORT_REQUEST_KEY, None)
    st.session_state.pop(READY_EXPORT_FILE_KEY, None)


def queue_export_request(unique_key, export_type, title=None):
    # Store which export the user wants. Generation happens after a loading state renders.
    st.session_state[PENDING_EXPORT_REQUEST_KEY] = {
        "unique_key": str(unique_key),
        "export_type": str(export_type),
        "title": title,
    }
    st.session_state.pop(READY_EXPORT_FILE_KEY, None)


def get_pending_export_request(unique_key):
    # Return pending export request for this answer only.
    request = st.session_state.get(PENDING_EXPORT_REQUEST_KEY)

    if not isinstance(request, dict):
        return None

    if str(request.get("unique_key")) != str(unique_key):
        return None

    return request


def get_ready_export_file(unique_key):
    # Return prepared export bytes for this answer only.
    prepared = st.session_state.get(READY_EXPORT_FILE_KEY)

    if not isinstance(prepared, dict):
        return None

    if str(prepared.get("unique_key")) != str(unique_key):
        return None

    return prepared


def display_export_loading(export_type):
    # Show loading before generating files like PDF/DOCX/XLSX/PPTX.
    st.markdown(
        f"""
<div class="export-loading-row">
    <div class="thinking-dots"><span></span><span></span><span></span></div>
    <span>Preparing {safe_html(export_type)}...</span>
</div>
""",
        unsafe_allow_html=True,
    )

    if EXPORT_FRONTEND_FLUSH_SECONDS > 0:
        time.sleep(EXPORT_FRONTEND_FLUSH_SECONDS)


def prepare_pending_export(answer, sources, unique_key, title=None):
    # Generate only the requested export type, not every format at once.
    request = get_pending_export_request(unique_key)

    if not request:
        return

    export_type = request.get("export_type")
    display_export_loading(export_type)

    data, filename, mime = get_export_file(
        answer=answer,
        sources=sources,
        export_type=export_type,
        title=request.get("title") or title,
    )

    if data is not None:
        st.session_state[READY_EXPORT_FILE_KEY] = {
            "unique_key": str(unique_key),
            "export_type": export_type,
            "data": data,
            "filename": filename,
            "mime": mime,
        }

    st.session_state.pop(PENDING_EXPORT_REQUEST_KEY, None)
    st.rerun()


def display_ready_export(unique_key):
    # Show the final download button after the file is prepared.
    prepared = get_ready_export_file(unique_key)

    if not prepared:
        return

    export_type = prepared.get("export_type") or "file"

    st.download_button(
        label=f"Download {export_type}",
        data=prepared.get("data"),
        file_name=prepared.get("filename"),
        mime=prepared.get("mime"),
        key=f"download_ready_{str(export_type).lower()}_{unique_key}",
        use_container_width=True,
    )


def display_export_buttons(answer, sources, unique_key, title=None):
    # Export panel is fast: it renders format buttons first.
    # The actual file is generated only after a specific format is clicked.
    options = [
        ("PDF", "PDF"),
        ("DOCX", "DOCX"),
        ("TXT", "TXT"),
        ("Markdown", "MD"),
        ("XLSX", "XLSX"),
        ("XLS", "XLS"),
        ("CSV", "CSV"),
        ("PPTX", "PPTX"),
        ("PPT", "PPT"),
    ]

    columns = st.columns(len(options))

    for column, option in zip(columns, options):
        export_type, label = option

        with column:
            st.button(
                label,
                key=f"prepare_export_{label.lower()}_{unique_key}",
                use_container_width=True,
                on_click=queue_export_request,
                args=(unique_key, export_type, title),
            )

    prepare_pending_export(answer, sources, unique_key, title=title)
    display_ready_export(unique_key)


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
]
