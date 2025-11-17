"""Vertex AI LLM helper (Gemini-style).

This assumes:
  - `google-cloud-aiplatform` is installed.
  - Application Default Credentials are configured.
  - Environment variables or parameters for project, location, and model are set.
"""
import os
from typing import List, Dict, Any

import vertexai
from vertexai.preview.generative_models import GenerativeModel

def build_rag_prompt(question: str, hits: List[Dict[str, Any]], top_n: int = 4) -> str:
    context_blocks = []
    for h in hits[:top_n]:
        meta = h.get("meta", {})
        page = meta.get("page", "?")
        source_type = meta.get("source_type", "unknown")
        prefix = f"[source={source_type}, page={page}]" if page != "?" else f"[source={source_type}]"
        context_blocks.append(f"{prefix} {h['text']}")
    context = "\n\n".join(context_blocks)

    prompt = f"""You are a precise credit policy assistant.

Use ONLY the context below to answer the question. If the answer is not clearly supported,
say you cannot answer based on the available policies/emails.

Question: {question}

Context:
{context}

Answer with:
  - A concise answer in 2-4 sentences.
  - Cite sources inline like [pdf p.X] or [email] when relevant.
"""
    return prompt

def generate_vertex_answer(
    question: str,
    hits: List[Dict[str, Any]],
    project_id: str | None = None,
    location: str | None = None,
    model_name: str | None = None,
) -> str:
    """Call Vertex AI LLM (Gemini-style) with a RAG prompt and return the text answer."""
    project_id = project_id or os.getenv("VERTEX_PROJECT_ID")
    location = location or os.getenv("VERTEX_LOCATION", "us-central1")
    model_name = model_name or os.getenv("VERTEX_MODEL_NAME", "gemini-1.5-pro")

    if not project_id:
        raise RuntimeError("VERTEX_PROJECT_ID not set")

    vertexai.init(project=project_id, location=location)
    model = GenerativeModel(model_name)

    prompt = build_rag_prompt(question, hits)
    response = model.generate_content(prompt)
    # GenerativeModel returns a response with candidates; text is usually in .text
    try:
        return response.text
    except Exception:
        return str(response)
