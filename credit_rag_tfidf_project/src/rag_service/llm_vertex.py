import os
import vertexai
from vertexai.preview.generative_models import GenerativeModel


def build_rag_prompt(question, hits, top_n: int = 5) -> str:
    context_lines = []
    for h in hits[:top_n]:
        meta = h.get("meta", {})
        src_type = meta.get("source_type", "doc")
        page = meta.get("page")
        prefix = f"[{src_type} p.{page}]" if page else f"[{src_type}]"
        context_lines.append(f"{prefix} {h['text']}")
    context = "\n\n".join(context_lines)

    return (
        "You are a helpful assistant answering questions about credit policies and emails.\n"
        "Use ONLY the provided context. If you are not sure, say you are not sure.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer concisely and, when possible, mention which source you used."
    )


def generate_vertex_answer(question, hits, project_id=None, location=None, model_name=None) -> str:
    project_id = project_id or os.getenv("VERTEX_PROJECT_ID")
    location = location or os.getenv("VERTEX_LOCATION", "us-central1")
    model_name = model_name or os.getenv("VERTEX_MODEL_NAME", "gemini-1.5-pro")

    if not project_id:
        return "Vertex project is not configured (VERTEX_PROJECT_ID missing)."

    vertexai.init(project=project_id, location=location)
    model = GenerativeModel(model_name)

    prompt = build_rag_prompt(question, hits)
    resp = model.generate_content(prompt)
    return resp.text
