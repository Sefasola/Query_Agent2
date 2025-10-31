import re

def normalize_for_compare(s: str) -> str:
    s = (s or "").strip()
    # sadeleştirme
    s = re.sub(r"\s+", " ", s)
    return s
