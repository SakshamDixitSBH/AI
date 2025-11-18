# Credit RAG Project (PDF + PST + Chroma + Vertex AI)

This project gives you a minimal but complete skeleton to:

1. Ingest PDF policy documents into Chroma.
2. Ingest Outlook PST emails (via `pypff`) into the same Chroma collection.
3. Search **PDF-only** or **email-only** from the command line.
4. Generate a Vertex AI LLM answer using retrieved chunks as context.
5. Expose a FastAPI REST endpoint for search (and optional RAG answer).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> Note: `pypff` (for PST parsing) may require native libraries (`libpff`) on your OS.
> If that's painful, you can first export PST to mbox/.eml and ingest those separately.

## Configuration

Edit `src/rag_service/config.py` if you want to change:
- Chroma persistence directory
- Collection name
- Embedding model (`all-MiniLM-L6-v2` by default)

For Vertex AI, set environment variables (or a `.env` file):

- `VERTEX_PROJECT_ID`
- `VERTEX_LOCATION` (e.g. `us-central1`)
- `VERTEX_MODEL_NAME` (e.g. `gemini-1.5-pro` or similar)

Make sure ADC (Application Default Credentials) are configured, e.g.:

```bash
gcloud auth application-default login
```

## 1. Ingest PDFs and PST emails

```bash
# Ingest a PDF
python -m scripts.ingest --pdf /path/to/Policy.pdf

# Ingest a PST archive of Outlook emails
python -m scripts.ingest --pst /path/to/Mailbox.pst
```

## 2. Search from command line

```bash
# PDF-only search
python -m scripts.search_pdf --query "utilization above 80%" --k 5

# Email-only search
python -m scripts.search_email --query "ACME Corp limit increase" --k 5
```

## 3. RAG answer with Vertex AI

```bash
python -m scripts.vertex_rag_answer \
    --question "Who can approve exposure between $50k and $250k?" \
   --kind all --k 5
```

This will:
- Retrieve top-k chunks from Chroma (PDF + email or filtered),
- Build a context-aware prompt,
- Call Vertex AI to generate an answer with citations.

## 4. REST API

Run the FastAPI app:

```bash
uvicorn rag_service.api:app --reload --port 8000
```

Then:

```bash
# Search endpoint
curl "http://localhost:8000/search?query=utilization+over+80%25&kind=pdf&k=5"

# RAG answer endpoint
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{"question":"Who approves exposure between 50k and 250k?","kind":"all","k":5}'
```
