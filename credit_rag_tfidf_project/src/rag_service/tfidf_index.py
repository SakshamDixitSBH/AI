from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfIndex:
    def __init__(self):
        self.vectorizer = None
        self.matrix = None
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        if not texts:
            return
        self.documents.extend(texts)
        self.metadatas.extend(metadatas)
        # Refit on all docs (fine for POC-scale)
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50000,
        )
        self.matrix = self.vectorizer.fit_transform(self.documents)

    def is_empty(self) -> bool:
        return not self.documents or self.vectorizer is None or self.matrix is None

    def search(self, query: str, kind: str = "all", k: int = 5):
        if self.is_empty():
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]

        # Sort all indices by score descending
        indices = scores.argsort()[::-1]

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


# Global singleton index
index = TfidfIndex()
