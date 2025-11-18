from chromadb.utils import embedding_functions
from .config import EMBEDDING_MODEL_NAME

def get_embedding_function():
    """Return a Chroma-compatible embedding function using SentenceTransformers."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
