from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader

# All file-type loaders are kept here to avoid too many loader files.
def load_txt(file_path):
    loader = TextLoader(
        file_path=str(file_path),
        encoding="utf-8",
        autodetect_encoding=True,
    )
    return loader.load()


# This is also used for .md and .markdown because they are text-based files.
def load_markdown(file_path):
    return load_txt(file_path)

# Loader for PDF files.
def load_pdf(file_path):
    loader = PyPDFLoader(
        file_path=str(file_path),
    )
    return loader.load()

# Loader for Word document files.
def load_docx(file_path):

    loader = UnstructuredWordDocumentLoader(
        file_path=str(file_path),
        mode="single",
    )
    return loader.load()


# Loader for CSV files.
def load_csv(file_path):

    loader = CSVLoader(
        file_path=str(file_path),
        encoding="utf-8",
        autodetect_encoding=True,
    )
    return loader.load()


# Loader for Excel files.
def load_xlsx(file_path):
    loader = UnstructuredExcelLoader(
        file_path=str(file_path),
        mode="single",
    )
    return loader.load()

# Loader for PowerPoint files.
def load_pptx(file_path):
    loader = UnstructuredPowerPointLoader(
        file_path=str(file_path),
        mode="single",
    )
    return loader.load()

# Map each file extension to the correct loader function.
LOADER_MAP = {
    ".txt": load_txt,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".csv": load_csv,
    ".xlsx": load_xlsx,
    ".xls": load_xlsx,
    ".pptx": load_pptx,
}

# Get the loader based on the file extension.
def get_loader(file_path):
    file_path = Path(file_path)
    return LOADER_MAP.get(file_path.suffix.lower())


# Used by tests to quickly show the supported file types.
def get_supported_extensions():
    return sorted(LOADER_MAP.keys())
