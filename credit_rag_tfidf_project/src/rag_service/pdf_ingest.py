from pathlib import Path
import re

import fitz  # PyMuPDF

from .tfidf_index import index


def ingest_pdf(pdf_path: str) -> int:
    pdf_path = str(Path(pdf_path).resolve())
    doc = fitz.open(pdf_path)

    texts = []
    metas = []

    for i, page in enumerate(doc):
        raw = page.get_text("text") or ""
        text = re.sub(r"\s+", " ", raw).strip()
        if not text:
            continue

        texts.append(text)
        metas.append(
            {
                "source_type": "pdf",
                "source": pdf_path,
                "page": i + 1,
            }
        )

    if texts:
        index.add_documents(texts, metas)

    return len(texts)
