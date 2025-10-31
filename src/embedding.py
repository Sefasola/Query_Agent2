import numpy as np
from sentence_transformers import SentenceTransformer

class E5Embedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        self.model = SentenceTransformer(model_name)

    def _ensure_list(self, texts):
        return texts if isinstance(texts, (list, tuple)) else [texts]

    def encode(self, texts) -> np.ndarray:
        texts = self._ensure_list(texts)
        emb = self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return emb.astype(np.float32)
