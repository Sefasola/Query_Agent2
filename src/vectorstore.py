from __future__ import annotations
from typing import List, Tuple
from langchain_chroma import Chroma
from langchain_core.documents import Document

def build_chroma(docs: List[Document], embeddings, persist_dir: str) -> Chroma:
    # Not: langchain-chroma 0.1.x ile persist() yok; persist_directory yeterli.
    vs = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="qa_multi_pdf",
    )
    return vs

def load_chroma(embeddings, persist_dir: str) -> Chroma:
    return Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_name="qa_multi_pdf",
    )

def to_documents(chunks) -> List[Document]:
    docs = []
    for ch in chunks:
        docs.append(Document(
            page_content=ch.text,
            metadata={"source": ch.source, "page": ch.page}
        ))
    return docs

def format_context(docs: List[Document], max_chars: int = 6000) -> str:
    parts = []
    size = 0
    for d in docs:
        header = f"[{d.metadata.get('source')} | s.{d.metadata.get('page')}]"
        txt = f"{header}\n{d.page_content.strip()}\n"
        if size + len(txt) > max_chars:
            break
        parts.append(txt)
        size += len(txt)
    return "\n".join(parts)

def best_scores_info(results_with_scores: List[Tuple[Document, float]]) -> Tuple[float, float]:
    if not results_with_scores:
        return 0.0, 0.0
    scores = [float(s) for _, s in results_with_scores]
    top1 = max(scores)
    top5 = scores[min(4, len(scores)-1)]
    return top1, top5
