import argparse

from rag_service.pdf_ingest import ingest_pdf
from rag_service.msg_ingest import ingest_msg


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs and .msg emails into TF-IDF index")
    parser.add_argument("--pdf", help="Path to a single PDF file to ingest")
    parser.add_argument("--msg", help="Path to a .msg file or folder of .msg files to ingest")
    args = parser.parse_args()

    if not args.pdf and not args.msg:
        parser.error("Provide at least one of --pdf or --msg")

    if args.pdf:
        count = ingest_pdf(args.pdf)
        print(f"Ingested {count} PDF chunks from {args.pdf}")

    if args.msg:
        count = ingest_msg(args.msg)
        print(f"Ingested {count} .msg email chunks from {args.msg}")


if __name__ == "__main__":
    main()
