import textwrap

from .vertex_client import vertex_gen_ai


def build_rag_prompt(question, hits, top_n: int = 5) -> str:
    """
    Build a RAG prompt from the user question + top-k BM25 hits.
    """

    context_lines = []
    for h in hits[:top_n]:
        meta = h.get("meta", {})
        src_type = meta.get("source_type", "doc")
        page = meta.get("page")
        if page is not None:
            prefix = f"[{src_type} p.{page}]"
        else:
            prefix = f"[{src_type}]"

        snippet = h["text"].strip().replace("\n", " ")
        context_lines.append(f"{prefix} {snippet}")

    context = "\n\n".join(context_lines)

    prompt = f"""
    You are a helpful assistant for credit policy and credit approval questions.

    Use ONLY the context below to answer. If the answer is not clearly present,
    say that you are not sure and indicate which document/page should be checked.

    Question:
    {question}

    Context:
    {context}

    Answer:
    """
    return textwrap.dedent(prompt).strip()


def generate_vertex_answer(question, hits, **kwargs) -> str:
    """
    Main LLM call used by FastAPI `/search` endpoint.

    Steps:
      1. Build RAG prompt from question + hits
      2. Send prompt to Vertex via VertexGenAI
      3. Return the model's text answer

    If anything fails, returns a fallback text + top hits for debugging.
    """

    if not hits:
        return "No relevant context found in the ingested documents."

    prompt = build_rag_prompt(question, hits)

    try:
        return vertex_gen_ai.generate(prompt)
    except Exception as e:
        # Fallback for POC – you can improve this later.
        lines = [
            f"Error calling Vertex: {e}",
            "",
            "Top hits (for debugging):",
        ]
        for h in hits[:3]:
            meta = h.get("meta", {})
            src_type = meta.get("source_type", "doc")
            page = meta.get("page")
            if page is not None:
                prefix = f"[{src_type} p.{page}]"
            else:
                prefix = f"[{src_type}]"
            snippet = h["text"][:200].replace("\n", " ")
            lines.append(f"- {prefix} {snippet}...")
        return "\n".join(lines)
