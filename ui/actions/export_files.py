from ui.actions.common import (
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
)



def export_txt(answer, sources=None, title=None):
    return build_export_text(answer=answer, sources=sources, title=title).encode("utf-8")


def export_md(answer, sources=None, title=None):
    lines = [
        f"# {title or 'InknowVa Response'}",
        "",
        normalize_answer_text(answer) or "No answer.",
    ]

    return "\n".join(lines).encode("utf-8")


def export_csv(answer, sources=None, title=None):
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Title", "Answer"])
    writer.writerow([str(title or "InknowVa Response"), normalize_answer_text(answer)])

    return output.getvalue().encode("utf-8-sig")


def export_xlsx(answer, sources=None, title=None):
    if Workbook is None:
        return None

    file = BytesIO()
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Response"

    sheet["A1"] = str(title or "InknowVa Response")
    sheet["A3"] = "Answer"
    sheet["A4"] = normalize_answer_text(answer) or "No answer."

    sheet.column_dimensions["A"].width = 100

    if Font is not None:
        sheet["A1"].font = Font(bold=True, size=16)
        sheet["A3"].font = Font(bold=True, size=12)

    if Alignment is not None:
        sheet["A1"].alignment = Alignment(wrap_text=True, vertical="top")
        sheet["A4"].alignment = Alignment(wrap_text=True, vertical="top")

    workbook.save(file)
    file.seek(0)
    return file


def export_xls(answer, sources=None, title=None):
    rows = [
        [str(title or "InknowVa Response")],
        [""],
        ["Answer"],
        [normalize_answer_text(answer) or "No answer."],
    ]

    html_rows = []
    for row_index, row in enumerate(rows):
        tag = "th" if row_index in [0, 2] else "td"
        cells = "".join(
            f"<{tag} style='padding:8px;border:1px solid #cccccc;vertical-align:top;text-align:left;white-space:pre-wrap;'>{html.escape(str(cell))}</{tag}>"
            for cell in row
        )
        html_rows.append(f"<tr>{cells}</tr>")

    html_doc = "".join([
        "<html><head><meta charset='utf-8'></head><body>",
        "<table style='border-collapse:collapse;font-family:Arial,sans-serif;font-size:12px;width:100%;'>",
        "".join(html_rows),
        "</table>",
        "</body></html>",
    ])

    return html_doc.encode("utf-8")



def add_docx_paragraph(doc, line):
    # Add one line to DOCX while preserving bullets and numbered items.
    raw_line = str(line or "")

    if not raw_line.strip():
        doc.add_paragraph("")
        return

    if is_bullet_line(raw_line):
        doc.add_paragraph(strip_list_marker(raw_line), style="List Bullet")
        return

    if is_numbered_line(raw_line):
        doc.add_paragraph(strip_list_marker(raw_line), style="List Number")
        return

    paragraph = doc.add_paragraph(raw_line.strip())
    paragraph_format = paragraph.paragraph_format
    paragraph_format.space_after = DocxPt(6) if DocxPt else None


def export_docx(answer, sources=None, title=None):
    if Document is None:
        return None

    file = BytesIO()
    doc = Document()

    if DocxInches is not None:
        for section in doc.sections:
            section.top_margin = DocxInches(0.75)
            section.bottom_margin = DocxInches(0.75)
            section.left_margin = DocxInches(0.75)
            section.right_margin = DocxInches(0.75)

    heading = doc.add_heading(str(title or "InknowVa Response"), level=0)

    if WD_ALIGN_PARAGRAPH is not None:
        heading.alignment = WD_ALIGN_PARAGRAPH.CENTER

    answer_lines = split_answer_lines(answer)

    if answer_lines:
        for line in answer_lines:
            add_docx_paragraph(doc, line)
    else:
        doc.add_paragraph("No answer.")

    doc.save(file)
    file.seek(0)
    return file



def get_pdf_styles():
    # Build reportlab styles with safe fallbacks.
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="AnswerText",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="BulletText",
        parent=styles["BodyText"],
        leftIndent=16,
        firstLineIndent=-8,
        fontSize=10,
        leading=14,
        spaceAfter=5,
    ))
    return styles


def add_pdf_answer_lines(story, styles, answer):
    # Add answer lines to PDF while preserving simple list formatting.
    answer_lines = split_answer_lines(answer)

    if not answer_lines:
        story.append(Paragraph("No answer.", styles["AnswerText"]))
        return

    for line in answer_lines:
        raw_line = str(line or "")

        if not raw_line.strip():
            story.append(Spacer(1, 6))
            continue

        escaped = html.escape(strip_list_marker(raw_line) if (is_bullet_line(raw_line) or is_numbered_line(raw_line)) else raw_line.strip())

        if is_bullet_line(raw_line):
            story.append(Paragraph(f"• {escaped}", styles["BulletText"]))
        elif is_numbered_line(raw_line):
            story.append(Paragraph(escaped, styles["BulletText"]))
        else:
            story.append(Paragraph(escaped, styles["AnswerText"]))


def export_pdf(answer, sources=None, title=None):
    if None in [getSampleStyleSheet, Paragraph, SimpleDocTemplate, Spacer, ParagraphStyle, colors]:
        return None

    file = BytesIO()
    pdf = SimpleDocTemplate(
        file,
        pagesize=letter,
        rightMargin=42,
        leftMargin=42,
        topMargin=42,
        bottomMargin=42,
    )
    styles = get_pdf_styles()
    story = [
        Paragraph(html.escape(str(title or "InknowVa Response")), styles["Title"]),
        Spacer(1, 10),
    ]

    add_pdf_answer_lines(story, styles, answer)

    pdf.build(story)
    file.seek(0)
    return file



def split_text_for_slides(text, chunk_size=650):
    text = normalize_answer_text(text)

    if not text:
        return ["No answer."]

    chunks = []
    while text:
        chunk = text[:chunk_size]
        cut = max(chunk.rfind(". "), chunk.rfind("\n"), chunk.rfind(" "))

        if cut > 180:
            chunk = text[:cut + 1]

        chunks.append(chunk.strip())
        text = text[len(chunk):].strip()

    return chunks


def export_pptx(answer, sources=None, title=None):
    if Presentation is None:
        return None

    file = BytesIO()
    presentation = Presentation()

    title_slide_layout = presentation.slide_layouts[0]
    slide = presentation.slides.add_slide(title_slide_layout)
    slide.shapes.title.text = str(title or "InknowVa Response")
    slide.placeholders[1].text = "Generated by InknowVa"

    content_layout = presentation.slide_layouts[1]
    for index, chunk in enumerate(split_text_for_slides(answer), start=1):
        slide = presentation.slides.add_slide(content_layout)
        slide.shapes.title.text = "Answer" if index == 1 else f"Answer {index}"
        body = slide.placeholders[1]
        body.text = chunk

        if PptxPt is not None:
            for paragraph in body.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.font.size = PptxPt(16)

    presentation.save(file)
    file.seek(0)
    return file


def export_ppt(answer, sources=None, title=None):
    rows = [
        str(title or "InknowVa Response"),
        "",
        normalize_answer_text(answer) or "No answer.",
    ]

    # Legacy .ppt binary generation is not supported directly.
    # This HTML-based .ppt opens in PowerPoint/Office with a compatibility warning.
    body = "<br>".join(html.escape(row) for row in rows)
    return f"<html><body>{body}</body></html>".encode("utf-8")



def get_export_file(answer, sources, export_type, title=None):
    # Return file data, filename, and MIME.
    exporters = {
        "PDF": export_pdf,
        "DOCX": export_docx,
        "TXT": export_txt,
        "Markdown": export_md,
        "CSV": export_csv,
        "XLSX": export_xlsx,
        "XLS": export_xls,
        "PPTX": export_pptx,
        "PPT": export_ppt,
    }

    if export_type not in EXPORT_TYPES:
        return None, None, None

    exporter = exporters.get(export_type)

    if not exporter:
        return None, None, None

    data = exporter(answer, sources, title=title)

    if data is None:
        return None, None, None

    filename = make_export_filename(export_type, title=title)
    _extension, mime = EXPORT_TYPES[export_type]

    return data, filename, mime


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
]
