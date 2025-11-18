from typing import List, Dict, Any
from pathlib import Path
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfIndex:
    def __init__(self):
        # Where we store the index on disk (relative to src/rag_service)
        base_dir = Path(__file__).resolve().parent.parent
        store_dir = base_dir / "tfidf_store"
        store_dir.mkdir(exist_ok=True)

        self._docs_path = store_dir / "documents.pkl"
        self._meta_path = store_dir / "metadatas.pkl"
        self._vec_path = store_dir / "vectorizer.pkl"
        self._mat_path = store_dir / "matrix.pkl"

        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []

        # Try to load existing index from disk
        self._load()

    # ----------------- public API -----------------

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        if not texts:
            return

        self.documents.extend(texts)
        self.metadatas.extend(metadatas)

        # Refit on all docs (fine for POC scale)
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50000,
        )
        self.matrix = self.vectorizer.fit_transform(self.documents)

        # Persist to disk so other processes can use it
        self._save()

    def is_empty(self) -> bool:
        return not self.documents or self.vectorizer is None or self.matrix is None

    def search(self, query: str, kind: str = "all", k: int = 5):
        if self.is_empty():
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]

        indices = scores.argsort()[::-1]  # highest score first

        results = []
        for idx in indices:
            meta = self.metadatas[idx]
            if kind == "pdf" and meta.get("source_type") != "pdf":
                continue
            if kind == "email" and meta.get("source_type") != "email":
                continue

            results.append(
                {
                    "rank": len(results) + 1,
                    "text": self.documents[idx],
                    "meta": meta,
                    "score": float(scores[idx]),
                }
            )
            if len(results) >= k:
                break

        return results

    # ----------------- internal helpers -----------------

    def _save(self) -> None:
        with open(self._docs_path, "wb") as f:
            pickle.dump(self.documents, f)
        with open(self._meta_path, "wb") as f:
            pickle.dump(self.metadatas, f)
        with open(self._vec_path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(self._mat_path, "wb") as f:
            pickle.dump(self.matrix, f)

    def _load(self) -> None:
        if not (
            self._docs_path.exists()
            and self._meta_path.exists()
            and self._vec_path.exists()
            and self._mat_path.exists()
        ):
            return  # nothing saved yet

        try:
            with open(self._docs_path, "rb") as f:
                self.documents = pickle.load(f)
            with open(self._meta_path, "rb") as f:
                self.metadatas = pickle.load(f)
            with open(self._vec_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            with open(self._mat_path, "rb") as f:
                self.matrix = pickle.load(f)
        except Exception:
            # If anything goes wrong, just start empty (POC-friendly)
            self.vectorizer = None
            self.matrix = None
            self.documents = []
            self.metadatas = []


# Global singleton
index = TfidfIndex()
