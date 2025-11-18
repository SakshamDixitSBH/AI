import argparse
from rag_service.pdf_ingest import ingest_pdf
from rag_service.pst_ingest import ingest_pst

def main():
    ap = argparse.ArgumentParser(description="Ingest PDFs and PST emails into Chroma")
    ap.add_argument("--pdf", help="Path to a PDF file to ingest")
    ap.add_argument("--pst", help="Path to a PST file to ingest")
    args = ap.parse_args()

    if not args.pdf and not args.pst:
        ap.error("Provide at least --pdf or --pst")

    if args.pdf:
        count = ingest_pdf(args.pdf)
        print(f"Ingested {count} PDF chunks from {args.pdf}")

    if args.pst:
        count = ingest_pst(args.pst)
        print(f"Ingested {count} email chunks from {args.pst}")

if __name__ == "__main__":
    main()
