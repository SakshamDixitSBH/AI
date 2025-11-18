from fastapi import FastAPI
from pydantic import BaseModel

from .search_service import search
from .llm_vertex import generate_vertex_answer


app = FastAPI(title="Credit RAG BM25 + Vertex API")


class SearchRequest(BaseModel):
    question: str           # natural language question
    kind: str = "all"       # "all" | "pdf" | "email"
    k: int = 5              # top-k hits to use as context


@app.get("/health")
def health():
    return {"status": "ok"}


# Optional: hits-only endpoint for debugging BM25 results
@app.get("/search_hits")
def search_hits(query: str, kind: str = "all", k: int = 5):
    hits = search(query, kind=kind, k=k)
    return {"hits": hits}


# Main RAG endpoint: search + Vertex LLM answer
@app.post("/search")
def search_and_answer(req: SearchRequest):
    """
    POC RAG endpoint.

    1. BM25 search over PDFs + emails
    2. Build RAG prompt from top-k hits
    3. Call Vertex GenAI via VertexGenAI wrapper
    4. Return final answer + raw hits
    """
    hits = search(req.question, kind=req.kind, k=req.k)
    answer = generate_vertex_answer(req.question, hits)
    return {
        "answer": answer,
        "hits": hits,
    }
