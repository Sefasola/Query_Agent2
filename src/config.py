from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import os

try:
    import yaml
except Exception:
    yaml = None  # yaml kurulu değilse from_yaml kullanımı opsiyonel kalır


@dataclass
class Settings:
    # Modeller
    embed_model: str = "intfloat/multilingual-e5-base"
    qwen_model: str = "Qwen/Qwen3-4B-Instruct-2507"

    # Retrieval / chunk
    chunk_size: int = 1200
    chunk_overlap: int = 120
    top_k: int = 5
    mmr: bool = True
    mmr_lambda: float = 0.5

    # eşikler
    min_relevance: float = 0.08
    low_conf_gap: float = 0.0

    # Depolama
    chroma_dir: str = "storage/chroma"

    # ---- Yükleyiciler ----
    @classmethod
    def from_yaml(cls, yaml_path: Optional[str] = None) -> "Settings":
        """
        config/settings.yaml dosyasından ayarları okur.
        Şema (minimum):
        models:
          query: Qwen/Qwen3-4B-Instruct-2507
          # embed: intfloat/multilingual-e5-base   (opsiyonel)

        YAML yoksa veya alan bulunamazsa -> varsayılanları korur.
        Ortam değişkenleri her zaman en yüksek önceliğe sahiptir:
          QWEN_MODEL, EMBED_MODEL
        """
        inst = cls()

        # 1) YAML oku (varsa)
        if yaml_path and Path(yaml_path).is_file() and yaml is not None:
            data: Dict[str, Any] = (yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {})
            models = data.get("models", {}) or {}

            qwen_from_yaml = models.get("query")
            if isinstance(qwen_from_yaml, str) and qwen_from_yaml.strip():
                inst.qwen_model = qwen_from_yaml.strip()

            embed_from_yaml = models.get("embed")
            if isinstance(embed_from_yaml, str) and embed_from_yaml.strip():
                inst.embed_model = embed_from_yaml.strip()

            # İstersen burada başka alanları da (chunk_size, chroma_dir vs.) yaml'dan okuyabilirsin.

        # 2) ENV override (her zaman öncelikli)
        qwen_env = os.getenv("QWEN_MODEL")
        if qwen_env and qwen_env.strip():
            inst.qwen_model = qwen_env.strip()

        embed_env = os.getenv("EMBED_MODEL")
        if embed_env and embed_env.strip():
            inst.embed_model = embed_env.strip()

        # Diğer olası ENV’ler (opsiyonel örnekler)
        chroma_env = os.getenv("CHROMA_DIR")
        if chroma_env and chroma_env.strip():
            inst.chroma_dir = chroma_env.strip()

        topk_env = os.getenv("TOP_K")
        if topk_env and topk_env.isdigit():
            inst.top_k = int(topk_env)

        return inst
