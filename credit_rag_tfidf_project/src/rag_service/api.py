from fastapi import FastAPI
from pydantic import BaseModel

from .search_service import search
from .llm_vertex import generate_vertex_answer

app = FastAPI(title="Credit RAG TF-IDF API")


class AnswerRequest(BaseModel):
    question: str
    kind: str = "all"
    k: int = 5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/search")
def search_api(query: str, kind: str = "all", k: int = 5):
    hits = search(query, kind=kind, k=k)
    return {"hits": hits}


@app.post("/answer")
def answer_api(req: AnswerRequest):
    hits = search(req.question, kind=req.kind, k=req.k)
    answer = generate_vertex_answer(req.question, hits)
    return {"answer": answer, "hits": hits}
