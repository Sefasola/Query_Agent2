from dataclasses import dataclass

@dataclass
class Settings:
    embed_model: str = "intfloat/multilingual-e5-base"
    qwen_model: str = "Qwen/Qwen3-4B-Instruct-2507"

    chunk_size: int = 1200
    chunk_overlap: int = 120
    top_k: int = 5
    mmr: bool = True
    mmr_lambda: float = 0.5

    # daha yumuşak eşikler
    min_relevance: float = 0.08   # 0.15 -> 0.08
    low_conf_gap: float = 0.0     # MMR ile zaten çeşitlilik var

    chroma_dir: str = "storage/chroma"
