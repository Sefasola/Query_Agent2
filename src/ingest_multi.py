from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dataclasses import dataclass

@dataclass
class PageChunk:
    page: int
    text: str
    source: str  # filename

def read_pdf_pages(pdf_path: str) -> List[PageChunk]:
    p = Path(pdf_path)
    reader = PdfReader(str(p))
    out: List[PageChunk] = []
    for i, pg in enumerate(reader.pages):
        txt = (pg.extract_text() or "").strip()
        if not txt:
            continue
        out.append(PageChunk(page=i+1, text=txt, source=p.name))
    return out

def chunk_pages(pages: List[PageChunk], chunk_size=1200, chunk_overlap=120) -> List[PageChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks: List[PageChunk] = []
    for pg in pages:
        parts = splitter.split_text(pg.text)
        for part in parts:
            chunks.append(PageChunk(page=pg.page, text=part, source=pg.source))
    return chunks

def iter_pdfs(pdf_dir: str) -> Iterable[str]:
    for p in Path(pdf_dir).glob("*.pdf"):
        yield str(p)
