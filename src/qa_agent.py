from __future__ import annotations
from typing import List
from langchain_core.documents import Document

from .config import Settings
from .prompts_loader import get_prompt
from .embeddings import build_embeddings
from .ingest_multi import read_pdf_pages, chunk_pages, iter_pdfs
from .vectorstore import build_chroma, load_chroma, to_documents, format_context, best_scores_info
from .qwen_llm import QwenChat
from .utils import normalize_for_compare

class QueryAgent:
    def __init__(self, settings: Settings, prompt_yaml: str = "prompts/prompts.yaml"):
        self.cfg = settings
        self.prompt_yaml = prompt_yaml
        self.emb = build_embeddings(self.cfg.embed_model)
        self.llm = QwenChat(self.cfg.qwen_model)

    # ---------- Build index for a folder of PDFs ----------
    def build_index(self, pdf_dir: str):
        all_chunks = []
        for pdf in iter_pdfs(pdf_dir):
            pages = read_pdf_pages(pdf)
            chunks = chunk_pages(pages, self.cfg.chunk_size, self.cfg.chunk_overlap)
            all_chunks.extend(chunks)
        docs = to_documents(all_chunks)
        build_chroma(docs, self.emb, self.cfg.chroma_dir)

    def _get_vs(self):
        return load_chroma(self.emb, self.cfg.chroma_dir)

    # ---------- Ask ----------
    def ask(self, question: str, k: int | None = None) -> str:
        k = k or self.cfg.top_k
        vs = self._get_vs()

        # MMR retriever (LangChain 0.2+ -> invoke)
        retriever = vs.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        docs: List[Document] = retriever.invoke(question)
        if not docs:
            return "BELİRTİLMEMİŞ"

        # Bağlamı hazırla
        context = format_context(docs, max_chars=6000)
        print("\n--- DEBUG CONTEXT PREVIEW ---\n", context[:1200], "\n-----------------------------\n")

        # 1) Extractive deneme
        sys = get_prompt(self.prompt_yaml, "extractive_no_wrap", "system")
        usr_tpl = get_prompt(self.prompt_yaml, "extractive_no_wrap", "user_template")
        user = usr_tpl.format(question=question, context=context)
        raw = self.llm.chat(sys, user).strip()

        if not raw or raw.upper().startswith("BEL") or raw == "BELİRTİLMEMİŞ":
            return "BELİRTİLMEMİŞ"

        # 2) Verifier: bağlamda destek var mı?
        ver_sys = get_prompt(self.prompt_yaml, "verify_supported", "system")
        ver_usr = get_prompt(self.prompt_yaml, "verify_supported", "user_template").format(
            question=question, answer=raw, context=context
        )
        verdict = self.llm.chat(ver_sys, ver_usr).strip().upper()
        if verdict != "YES":
            if normalize_for_compare(raw) not in normalize_for_compare(context):
                return "BELİRTİLMEMİŞ"

        return " ".join(raw.split())
