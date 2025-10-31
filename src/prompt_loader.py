from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml

_CACHE: Dict[str, Dict[str, Any]] = {}

def _load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt YAML dosyası bulunamadı: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_prompts(path: str | Path = "prompts/prompts.yaml") -> Dict[str, Any]:
    key = str(Path(path).resolve())
    if key not in _CACHE:
        _CACHE[key] = _load_yaml(path)
    return _CACHE[key]

def get_prompt(dot_key: str, path: str | Path = "prompts/prompts.yaml") -> str:
    """
    dot_key ör.: 'extractive_tr.system'
    """
    data = load_prompts(path)
    cur: Any = data
    for part in dot_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Prompt anahtarı bulunamadı: {dot_key}")
        cur = cur[part]
    if not isinstance(cur, str) or not cur.strip():
        raise ValueError(f"Prompt boş/uygunsuz: {dot_key}")
    return cur
