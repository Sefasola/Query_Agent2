# src/reranker.py
from __future__ import annotations
from typing import List, Tuple
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

class Reranker:
    """
    BAAI/bge-reranker-base ile tekrar sıralama.
    threshold ~ 0.24-0.28 aralığı genelde güvenli.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", threshold: float = 0.25):
        self.model = CrossEncoder(model_name)
        self.threshold = float(threshold)

    def score(self, question: str, docs: List[Document]) -> List[Tuple[Document, float]]:
        pairs = [(question, d.page_content) for d in docs]
        scores = self.model.predict(pairs).tolist()
        ranked = list(sorted(zip(docs, scores), key=lambda x: x[1], reverse=True))
        return ranked

    def filter(self, question: str, docs: List[Document], top_n: int = 5) -> Tuple[List[Document], List[float]]:
        ranked = self.score(question, docs)
        keep_docs, keep_scores = [], []
        for d, s in ranked:
            if s >= self.threshold and len(keep_docs) < top_n:
                keep_docs.append(d); keep_scores.append(float(s))
        # eşik altıysa, ilk top_n'i yine de döndür (çok sert olmasın)
        if not keep_docs:
            ranked_docs = [d for d, _ in ranked[:top_n]]
            ranked_scores = [float(s) for _, s in ranked[:top_n]]
            return ranked_docs, ranked_scores
        return keep_docs, keep_scores
