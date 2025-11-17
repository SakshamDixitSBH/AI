import re
import uuid
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF

from .vector_store import get_collection

def _normalize_ws(t: str) -> str:
    t = t.replace("\xa0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s+\n", "\n", t)
    return t.strip()

def _read_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page": i + 1, "text": text})
    return pages

def _split_into_paragraphs(text: str) -> List[str]:
    text = _normalize_ws(text)
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    return paras if paras else ([text] if text else [])

def _sliding_chunks(paragraphs: List[str], target_tokens=180, overlap_tokens=40) -> List[str]:
    words = []
    for p in paragraphs:
        words.extend(p.split())
        words.append("<PBRK>")
    chunks, i, n = [], 0, len(words)
    while i < n:
        j = min(n, i + target_tokens)
        k = j
        while k > i and k < n and words[k - 1] != "<PBRK>":
            k -= 1
        if k <= i + target_tokens * 0.5:
            k = j
        seg = [w for w in words[i:k] if w != "<PBRK>"]
        txt = " ".join(seg).strip()
        if txt:
            chunks.append(txt)
        if k >= n:
            break
        i = max(k - overlap_tokens, 0)
        if i == k:
            i += 1
    return chunks

def ingest_pdf(pdf_path: str) -> int:
    """Ingest a PDF into Chroma as source_type='pdf'. Returns number of chunks."""
    col = get_collection()
    pdf_path = str(Path(pdf_path).resolve())
    pages = _read_pdf_pages(pdf_path)

    ids, docs, metas = [], [], []
    for p in pages:
        paras = _split_into_paragraphs(p["text"])
        chunks = _sliding_chunks(paras, target_tokens=180, overlap_tokens=40)
        for ci, chunk in enumerate(chunks):
            ids.append(str(uuid.uuid4()))
            docs.append(chunk)
            metas.append({
                "source_type": "pdf",
                "source": pdf_path,
                "page": p["page"],
                "chunk": ci,
            })

    if docs:
        col.add(ids=ids, documents=docs, metadatas=metas)
    return len(docs)
