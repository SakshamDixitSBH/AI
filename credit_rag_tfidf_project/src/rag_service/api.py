from fastapi import FastAPI
from pydantic import BaseModel

from .search_service import search
from .llm_vertex import generate_vertex_answer


app = FastAPI(title="Credit RAG BM25 API")


class SearchRequest(BaseModel):
    question: str
    kind: str = "all"      # "all", "pdf", or "email"
    k: int = 5             # top-k hits to use for LLM prompt


@app.get("/health")
def health():
    return {"status": "ok"}


# OPTIONAL: hits-only endpoint for debugging BM25 ranking
@app.get("/search_hits")
def search_hits(query: str, kind: str = "all", k: int = 5):
    hits = search(query, kind=kind, k=k)
    return {"hits": hits}


# MAIN RAG ENDPOINT (search + LLM answer)
@app.post("/search")
def search_and_answer(req: SearchRequest):
    """
    This endpoint now performs:
    1. BM25 search
    2. Builds prompt
    3. Calls LLM (or stub)
    4. Returns final LLM answer + raw hits
    """
    hits = search(req.question, kind=req.kind, k=req.k)
    answer = generate_vertex_answer(req.question, hits)

    return {
        "answer": answer,
        "hits": hits
    }
