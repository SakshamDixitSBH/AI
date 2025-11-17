import chromadb
from .config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from .embeddings import get_embedding_function

_client = None
_collection = None

def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _client

def get_collection():
    global _collection
    if _collection is None:
        client = get_client()
        emb_fn = get_embedding_function()
        _collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=emb_fn,
        )
    return _collection
