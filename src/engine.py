from pathlib import Path
import numpy as np
import json
from .ingest import pdf_to_pages, save_pages, load_pages
from .embedding import E5Embedder
from .qwen_client import QwenClient
from .utils import parse_extractive_answer, is_verbatim_substring
from .prompt_loader import get_prompt

class QAPipeline:
    def __init__(
        self,
        pdf_path: str | None = None,
        storage_dir: str = "storage",
        prompt_yaml_path: str = "prompts/prompts.yaml",
        prompt_key: str = "extractive_tr.system",
    ):
        self.pdf_path = pdf_path
        self.storage_dir = Path(storage_dir)
        self.pages_json = self.storage_dir / "pages.json"
        self.emb_npy = self.storage_dir / "embeddings.npy"
        self.embedder = E5Embedder()

        system_prompt = get_prompt(prompt_key, prompt_yaml_path)
        self.llm = QwenClient(system_prompt=system_prompt)

    # ---------- Build / Load ----------
    def build_index(self):
        assert self.pdf_path, "PDF yolu verilmemiş."
        pages = pdf_to_pages(self.pdf_path)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        save_pages(pages, str(self.pages_json))

        texts = [p["text"] for p in pages]
        emb = self.embedder.encode(texts)   # (N, D) normalized
        np.save(self.emb_npy, emb)

    def _ensure_loaded(self):
        if not self.pages_json.exists() or not self.emb_npy.exists():
            if not self.pdf_path:
                raise RuntimeError("İndeks yok. Lütfen --pdf ile --build çalıştırın.")
            self.build_index()
        self.pages = load_pages(str(self.pages_json))
        self.emb = np.load(self.emb_npy)  # normalized

    # ---------- Retrieval ----------
    def _search_pages(self, query: str, top_k: int = 3):
        q = self.embedder.encode(query)[0]    # (D,)
        scores = self.emb @ q                 # cosine ~ dot (normalized)
        idx = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in idx]

    # ---------- Ask ----------
    def ask(self, question: str, top_k: int = 3) -> str:
        self._ensure_loaded()
        hits = self._search_pages(question, top_k=top_k)

        best_answer = None
        best_len = 10**9

        for i, _ in hits:
            page = self.pages[i]
            raw_out = self.llm.generate(page_text=page["text"], question=question)

            if raw_out.strip() == "BELİRTİLMEMİŞ":
                continue

            ans = parse_extractive_answer(raw_out)
            if not ans:
                continue

            if not is_verbatim_substring(ans, page["text"]):
                continue

            if len(ans) < best_len:
                best_len = len(ans)
                best_answer = ans

        return f"<<<{best_answer}>>>" if best_answer else "BELİRTİLMEMİŞ"
