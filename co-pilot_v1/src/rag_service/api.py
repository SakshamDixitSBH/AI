from typing import Literal, List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .search_service import search, Kind
from .llm_vertex import generate_vertex_answer

app = FastAPI(title="Credit RAG Search API", version="0.1.0")

class SearchResponseItem(BaseModel):
    rank: int
    distance: float
    text: str
    meta: Dict[str, Any]

class SearchResponse(BaseModel):
    query: str
    kind: Kind
    k: int
    results: List[SearchResponseItem]

@app.get("/search", response_model=SearchResponse)
def search_endpoint(query: str, kind: Kind = "all", k: int = 5):
    hits = search(query, kind=kind, k=k)
    return SearchResponse(
        query=query,
        kind=kind,
        k=k,
        results=[SearchResponseItem(**h) for h in hits],
    )

class AnswerRequest(BaseModel):
    question: str
    kind: Kind = "all"
    k: int = 5
    project_id: Optional[str] = None
    location: Optional[str] = None
    model_name: Optional[str] = None

class AnswerResponse(BaseModel):
    question: str
    kind: Kind
    k: int
    answer: str
    results: List[SearchResponseItem]

@app.post("/answer", response_model=AnswerResponse)
def answer_endpoint(req: AnswerRequest):
    hits = search(req.question, kind=req.kind, k=req.k)
    if not hits:
        raise HTTPException(status_code=404, detail="No context found for question")

    answer = generate_vertex_answer(
        req.question,
        hits,
        project_id=req.project_id,
        location=req.location,
        model_name=req.model_name,
    )

    return AnswerResponse(
        question=req.question,
        kind=req.kind,
        k=req.k,
        answer=answer,
        results=[SearchResponseItem(**h) for h in hits],
    )
