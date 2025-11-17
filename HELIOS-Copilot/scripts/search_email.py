import argparse
from rag_service.search_service import search

def main():
    ap = argparse.ArgumentParser(description="Search email-only content in Chroma")
    ap.add_argument("--query", required=True, help="Search query text")
    ap.add_argument("--k", type=int, default=5, help="Number of results")
    args = ap.parse_args()

    hits = search(args.query, kind="email", k=args.k)
    if not hits:
        print("No results.")
        return

    for h in hits:
        meta = h["meta"]
        subject = meta.get("subject", "(no subject)")
        sent_at = meta.get("sent_at", "")
        print(f"[{h['rank']}] dist={h['distance']:.4f} subject={subject} sent_at={sent_at}")
        snippet = h["text"].replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:300].rstrip() + "..."
        print(snippet)
        print("-" * 80)

if __name__ == "__main__":
    main()
