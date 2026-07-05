from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_REVISION,
    EMBEDDING_NORMALIZE,
    
)


def get_embedding_model(
    model_name=None,
    model_revision=None,
    device=None,
    normalize_embeddings=None,
    batch_size=None,
):
    # Create a LangChain-compatible embedding model.
    selected_model_name = model_name or EMBEDDING_MODEL_NAME
    selected_model_revision = model_revision or EMBEDDING_MODEL_REVISION
    selected_device = device or EMBEDDING_DEVICE
    selected_normalize = EMBEDDING_NORMALIZE if normalize_embeddings is None else normalize_embeddings
    selected_batch_size = batch_size or EMBEDDING_BATCH_SIZE

    return HuggingFaceEmbeddings(
        model_name=selected_model_name,
        model_kwargs={
            "device": selected_device,
            "revision": selected_model_revision,
        },
        encode_kwargs={
            "normalize_embeddings": selected_normalize,
            "batch_size": selected_batch_size,
        },
    )
