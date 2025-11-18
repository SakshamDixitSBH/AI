import argparse
from rag_service.search_service import search


def main():
    parser = argparse.ArgumentParser(description="Search only in PDF content")
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    hits = search(args.query, kind="pdf", k=args.k)
    for h in hits:
        meta = h["meta"]
        print(f"RANK {h['rank']} | SCORE {h['score']:.4f} | PAGE {meta.get('page')} | SOURCE {meta.get('source')}")
        print(h["text"][:300].replace("\n", " "))
        print("-" * 80)


if __name__ == "__main__":
    main()
