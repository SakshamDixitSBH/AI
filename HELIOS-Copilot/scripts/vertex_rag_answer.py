import argparse
from rag_service.search_service import search
from rag_service.llm_vertex import generate_vertex_answer

def main():
    ap = argparse.ArgumentParser(description="RAG answer using Chroma + Vertex AI")
    ap.add_argument("--question", required=True, help="User question")
    ap.add_argument("--kind", choices=["all", "pdf", "email"], default="all", help="Source filter")
    ap.add_argument("--k", type=int, default=5, help="Top-k results for retrieval")
    ap.add_argument("--project_id", help="GCP project ID (optional, else env var)")
    ap.add_argument("--location", help="Vertex location (optional, else env var)")
    ap.add_argument("--model_name", help="Vertex model name (optional, else env var)")
    args = ap.parse_args()

    hits = search(args.question, kind=args.kind, k=args.k)
    if not hits:
        print("No context found.")
        return

    answer = generate_vertex_answer(
        args.question,
        hits,
        project_id=args.project_id,
        location=args.location,
        model_name=args.model_name,
    )

    print("=== Answer ===")
    print(answer)
    print("\n=== Top Context Chunks ===")
    for h in hits:
        meta = h["meta"]
        source_type = meta.get("source_type", "?")
        page = meta.get("page", "")
        label = f"{source_type} p.{page}" if page else source_type
        print(f"[{h['rank']}] {label} dist={h['distance']:.4f}")
        snippet = h["text"].replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200].rstrip() + "..."
        print(snippet)
        print("-" * 60)

if __name__ == "__main__":
    main()
