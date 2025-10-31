from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml

_cache: Dict[str, Dict[str, Any]] = {}

def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def get_prompt(path: str, section: str, key: str) -> str:
    # section: extractive_no_wrap | verify_supported
    # key: system | user_template
    global _cache
    abspath = str(Path(path).resolve())
    if abspath not in _cache:
        _cache[abspath] = load_yaml(path)
    data = _cache[abspath]
    try:
        out = data[section][key]
    except Exception:
        raise KeyError(f"Prompt bulunamadı: {section}.{key} in {path}")
    if not isinstance(out, str) or not out.strip():
        raise ValueError(f"Prompt boş: {section}.{key}")
    return out
