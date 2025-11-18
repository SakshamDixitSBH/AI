from typing import List, Dict, Any
from pathlib import Path
import pickle
import numpy as np


def _tokenize(text: str) -> List[str]:
    # Super simple tokenizer; you can later improve (regex, stopwords, etc.)
    return text.lower().split()


def _bm25_score(
    query_terms: List[str],
    doc_terms: List[str],
    term_freq: Dict[str, int],
    doc_len: int,
    avgdl: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """
    Standard BM25 scoring for one document.
    """
    score = 0.0
    if avgdl == 0:
        return score

    # Precompute term frequencies in query to avoid overweighting repeated words
    q_terms = {}
    for t in query_terms:
        q_terms[t] = q_terms.get(t, 0) + 1

    for term, qf in q_terms.items():
        if term not in term_freq:
            continue

        f = term_freq[term]
        # Classic BM25, IDF skipped for simplicity (in practice still works very well).
        # You can add proper IDF if you maintain DF stats.
        numerator = f * (k1 + 1)
        denominator = f + k1 * (1.0 - b + b * (doc_len / avgdl))
        score += (numerator / denominator)

    return score


class TfidfIndex:
    """
    BM25-based index (name kept for backward compatibility with earlier TF-IDF version).
    Stores:
      - documents: list of raw text
      - metadatas: list of dicts
      - doc_lengths: word counts
      - avgdl: average document length
    All persisted to disk so ingestion and search can run in different processes.
    """

    def __init__(self):
        base_dir = Path(__file__).resolve().parent.parent
        store_dir = base_dir / "tfidf_store"
        store_dir.mkdir(exist_ok=True)

        self._docs_path = store_dir / "documents.pkl"
        self._meta_path = store_dir / "metadatas.pkl"
        self._lens_path = store_dir / "doc_lengths.pkl"

        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0

        self._load()

    # ----------------- public API -----------------

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """
        Add a batch of documents (chunks) to the index and recompute statistics.
        """
        if not texts:
            return

        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must have the same length")

        self.documents.extend(texts)
        self.metadatas.extend(metadatas)

        # Recompute lengths and avgdl
        self.doc_lengths = [len(_tokenize(t)) for t in self.documents]
        if self.doc_lengths:
            self.avgdl = float(np.mean(self.doc_lengths))
        else:
            self.avgdl = 0.0

        self._save()

    def is_empty(self) -> bool:
        return not self.documents

    def search(self, query: str, kind: str = "all", k: int = 5):
        """
        BM25 search.
        kind: "all" | "pdf" | "email"
        Returns list of dicts:
        {
          "rank": int,
          "text": str,
          "meta": {...},
          "score": float,      # BM25 score (higher is better)
          "distance": float,   # derived (lower is better; kept for compatibility)
        }
        """
        if self.is_empty() or not query.strip():
            return []

        query_terms = _tokenize(query)

        scores = []
        for idx, text in enumerate(self.documents):
            terms = _tokenize(text)
            term_freq: Dict[str, int] = {}
            for t in terms:
                term_freq[t] = term_freq.get(t, 0) + 1

            doc_len = self.doc_lengths[idx] if idx < len(self.doc_lengths) else len(terms)
            s = _bm25_score(query_terms, terms, term_freq, doc_len, self.avgdl)
            scores.append(s)

        scores = np.array(scores)
        indices = scores.argsort()[::-1]  # highest score first

        results = []
        for idx in indices:
            meta = self.metadatas[idx]

            if kind == "pdf" and meta.get("source_type") != "pdf":
                continue
            if kind == "email" and meta.get("source_type") != "email":
                continue

            score = float(scores[idx])

            # distance: lower is better; we map BM25 score to (0,1] via 1/(1+score)
            distance = float(1.0 / (1.0 + max(score, 0.0)))

            results.append(
                {
                    "rank": len(results) + 1,
                    "text": self.documents[idx],
                    "meta": meta,
                    "score": score,
                    "distance": distance,
                }
            )
            if len(results) >= k:
                break

        return results

    # ----------------- persistence helpers -----------------

    def _save(self) -> None:
        with open(self._docs_path, "wb") as f:
            pickle.dump(self.documents, f)
        with open(self._meta_path, "wb") as f:
            pickle.dump(self.metadatas, f)
        with open(self._lens_path, "wb") as f:
            pickle.dump({"doc_lengths": self.doc_lengths, "avgdl": self.avgdl}, f)

    def _load(self) -> None:
        if not (self._docs_path.exists() and self._meta_path.exists() and self._lens_path.exists()):
            # No existing index; start empty
            return

        try:
            with open(self._docs_path, "rb") as f:
                self.documents = pickle.load(f)
            with open(self._meta_path, "rb") as f:
                self.metadatas = pickle.load(f)
            with open(self._lens_path, "rb") as f:
                stats = pickle.load(f)
                self.doc_lengths = stats.get("doc_lengths", [])
                self.avgdl = float(stats.get("avgdl", 0.0))
        except Exception:
            # If anything goes wrong, start clean (POC-friendly)
            self.documents = []
            self.metadatas = []
            self.doc_lengths = []
            self.avgdl = 0.0


# Global singleton index
index = TfidfIndex()
