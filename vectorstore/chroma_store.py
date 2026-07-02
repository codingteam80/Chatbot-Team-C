import shutil
from pathlib import Path

from langchain_chroma import Chroma

from config.settings import CHROMA_COLLECTION_NAME, CHROMA_PATH


def has_chroma_files(persist_directory=CHROMA_PATH):
    # Check kung may existing ChromaDB folder na may laman.
    path = Path(persist_directory)

    if not path.exists():
        return False

    if not path.is_dir():
        return False

    return any(path.iterdir())


def reset_chroma_folder(persist_directory=CHROMA_PATH):
    # Burahin ang old ChromaDB folder bago gumawa ng bagong vectors.
    # Ginagamit ito sa full re-ingest para maiwasan ang duplicate vectors.
    path = Path(persist_directory)

    if path.exists():
        shutil.rmtree(path)

    print(f"[CHROMA] Reset folder: {persist_directory}", flush=True)


def create_chroma_vectorstore(
    chunks,
    embedding_model,
    persist_directory=CHROMA_PATH,
    collection_name=CHROMA_COLLECTION_NAME,
):
    # Gumawa ng bagong ChromaDB mula sa chunks.
    # Dito nangyayari ang actual embedding + save sa Chroma.
    if not chunks:
        raise ValueError("No chunks received. Check load -> clean -> chunk steps first.")

    if embedding_model is None:
        raise ValueError("No embedding model received. Check get_embedding_model().")

    print(f"[CHROMA] Creating vectorstore from {len(chunks)} chunks...", flush=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    print(f"[CHROMA] Vectorstore saved to: {persist_directory}", flush=True)

    return vectorstore


def load_chroma_vectorstore(
    embedding_model,
    persist_directory=CHROMA_PATH,
    collection_name=CHROMA_COLLECTION_NAME,
):
    # I-load ang existing ChromaDB para magamit sa retrieval/chatbot.
    path = Path(persist_directory)

    if not path.exists():
        raise FileNotFoundError(f"ChromaDB folder not found: {persist_directory}. Run ingest.py first.")

    if not path.is_dir():
        raise NotADirectoryError(f"ChromaDB path is not a folder: {persist_directory}")

    if not any(path.iterdir()):
        raise FileNotFoundError(f"ChromaDB folder is empty: {persist_directory}. Run ingest.py first.")

    if embedding_model is None:
        raise ValueError("No embedding model received. Check get_embedding_model().")

    print(f"[CHROMA] Loading vectorstore from: {persist_directory}", flush=True)

    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_name=collection_name,
    )


def get_chroma_document_count(vectorstore):
    # Kunin kung ilang vectors/documents ang nasa Chroma collection.
    if vectorstore is None:
        return 0

    return vectorstore._collection.count()


def print_chroma_status(persist_directory=CHROMA_PATH):
    # Optional helper para madaling makita kung may ChromaDB na.
    if has_chroma_files(persist_directory):
        print(f"[CHROMA] Existing ChromaDB found: {persist_directory}", flush=True)
    else:
        print(f"[CHROMA] No ChromaDB found yet: {persist_directory}", flush=True)


if __name__ == "__main__":
    # Pang quick check lang.
    # Hindi ito ang normal way ng ingest.
    # Normal run dapat: python ingest.py
    print_chroma_status()
