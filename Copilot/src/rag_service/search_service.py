from typing import Literal, List, Dict, Any
from .vector_store import get_collection

Kind = Literal["all", "pdf", "email"]

def search(query: str, kind: Kind = "all", k: int = 5) -> List[Dict[str, Any]]:
    """Search the Chroma collection, optionally filtering by source_type."""
    col = get_collection()
    where = None
    if kind == "pdf":
        where = {"source_type": "pdf"}
    elif kind == "email":
        where = {"source_type": "email"}

    result = col.query(
        query_texts=[query],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    hits: List[Dict[str, Any]] = []
    if not result["documents"] or not result["documents"][0]:
        return hits

    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        hits.append({
            "rank": i,
            "distance": float(dist),
            "text": doc,
            "meta": meta,
        })
    return hits
