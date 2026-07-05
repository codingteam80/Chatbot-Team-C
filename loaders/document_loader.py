from pathlib import Path
from loaders.file_loaders import get_loader, get_supported_extensions

EMPTY_REPORT = {
    "loaded_files": 0,
    "loaded_docs": 0,
    "skipped_files": [],
    "failed_files": [],
    "loaded_file_details": [],
}


def add_standard_metadata(docs, file_path):
    # Add common metadata for consistent retrieval and source UI display.
    file_path = Path(file_path)

    for doc_index, doc in enumerate(docs):
        doc.metadata = dict(doc.metadata or {})
        doc.metadata.setdefault("source", str(file_path))
        doc.metadata["file_name"] = file_path.name
        doc.metadata["file_stem"] = file_path.stem
        doc.metadata["file_ext"] = file_path.suffix.lower()
        doc.metadata["document_index"] = doc_index

    return docs


def is_valid_document(doc):
    # Keep only documents with readable text.
    text = getattr(doc, "page_content", "")
    return bool(str(text or "").strip())


def create_report(data_path, recursive):
    # Report which files were loaded, skipped, and failed.
    report = dict(EMPTY_REPORT)
    report["data_path"] = str(data_path)
    report["recursive"] = recursive
    report["supported_extensions"] = get_supported_extensions()
    report["skipped_files"] = []
    report["failed_files"] = []
    report["loaded_file_details"] = []
    return report


def iter_source_files(data_path, recursive=True):
    # Get files from the data folder in stable order.
    data_path = Path(data_path)
    files = data_path.rglob("*") if recursive else data_path.iterdir()

    for file_path in sorted(files):
        if file_path.is_file():
            yield file_path


def validate_data_path(data_path):
    # Make sure the data path is an existing folder.
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data folder not found: {data_path}")

    if not data_path.is_dir():
        raise NotADirectoryError(f"Data path is not a folder: {data_path}")

    return data_path


def load_single_file(file_path):
    # Load one supported file and return valid docs.
    loader_func = get_loader(file_path)

    if loader_func is None:
        return None

    docs = loader_func(file_path)
    docs = [doc for doc in docs if is_valid_document(doc)]
    return add_standard_metadata(docs, file_path)


def load_documents(data_path, recursive=True, return_report=False):
    # Main document loader for ingest, tests, and app usage.
    data_path = validate_data_path(data_path)
    documents = []
    report = create_report(data_path, recursive)

    for file_path in iter_source_files(data_path, recursive=recursive):
        loader_func = get_loader(file_path)

        if loader_func is None:
            report["skipped_files"].append({
                "file_path": str(file_path),
                "reason": "Unsupported file type",
            })
            continue

        try:
            docs = loader_func(file_path)
            docs = [doc for doc in docs if is_valid_document(doc)]
            docs = add_standard_metadata(docs, file_path)

            if not docs:
                report["skipped_files"].append({
                    "file_path": str(file_path),
                    "reason": "No readable text found",
                })
                continue

            documents.extend(docs)
            report["loaded_files"] += 1
            report["loaded_docs"] += len(docs)
            report["loaded_file_details"].append({
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_ext": file_path.suffix.lower(),
                "docs_loaded": len(docs),
            })

        except Exception as error:
            # Skip broken files, but continue loading the other files.
            report["failed_files"].append({
                "file_path": str(file_path),
                "error_type": type(error).__name__,
                "error_message": str(error),
            })

    if return_report:
        return documents, report

    return documents
