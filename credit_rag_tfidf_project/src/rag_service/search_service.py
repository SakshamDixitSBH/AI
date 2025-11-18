from .tfidf_index import index


def search(query: str, kind: str = "all", k: int = 5):
    return index.search(query, kind=kind, k=k)
