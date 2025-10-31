from pypdf import PdfReader
from pathlib import Path
import json
from .utils import clean_text

def pdf_to_pages(pdf_path: str) -> list[dict]:
    reader = PdfReader(pdf_path)
    pages = []
    for i, pg in enumerate(reader.pages):
        raw = pg.extract_text() or ""
        txt = clean_text(raw)
        pages.append({"page": i+1, "text": txt})
    return pages

def save_pages(pages: list[dict], out_json: str):
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

def load_pages(in_json: str) -> list[dict]:
    with open(in_json, "r", encoding="utf-8") as f:
        return json.load(f)
