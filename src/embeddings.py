from __future__ import annotations
from typing import List
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

class E5Embeddings(Embeddings):
    """
    E5 için doğru kullanım:
      - Belgeler:  'passage: {text}'
      - Sorgular:  'query: {question}'
    """
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", normalize: bool = True):
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [f"passage: {t}" for t in texts]
        vecs = self.model.encode(
            texts, normalize_embeddings=self.normalize,
            convert_to_numpy=True, show_progress_bar=False
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> List[float]:
        text = f"query: {text}"
        vec = self.model.encode(
            [text], normalize_embeddings=self.normalize,
            convert_to_numpy=True, show_progress_bar=False
        )[0]
        return vec.tolist()

def build_embeddings(model_name: str = "intfloat/multilingual-e5-base") -> E5Embeddings:
    return E5Embeddings(model_name=model_name, normalize=True)
