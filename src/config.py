from dataclasses import dataclass

@dataclass
class Settings:
    # Embedding model
    embed_model: str = "intfloat/multilingual-e5-base"

    # Qwen chat model
    qwen_model: str = "Qwen/Qwen3-8B-Instruct"

    # Retrieval
    chunk_size: int = 1200
    chunk_overlap: int = 120
    top_k: int = 5
    mmr: bool = True
    mmr_lambda: float = 0.5

    # Gating thresholds
    min_relevance: float = 0.15   # altı ise BELİRTİLMEMİŞ
    low_conf_gap: float = 0.06    # (top1 - top5) küçükse riskli kabul

    # Paths
    chroma_dir: str = "storage/chroma"
