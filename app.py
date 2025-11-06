# app.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from rich import print as rprint

from src.config import Settings
from src.qa_agent import QueryAgent
from src.schemas import OutputSchema


# -------------------- Yardımcılar --------------------

def _load_models_from_settings(settings_path: Optional[str], cfg: Settings) -> None:
    """
    config/settings.yaml içinden model adlarını okumak için.
    Format:
    models:
      query: Qwen/Qwen3-4B-Instruct-2507
    """
    if not settings_path:
        return
    p = Path(settings_path)
    if not p.exists():
        rprint(f"[yellow]Uyarı:[/yellow] Ayar dosyası bulunamadı: {settings_path}")
        return

    try:
        import yaml  # type: ignore
    except Exception:
        rprint("[yellow]Uyarı:[/yellow] PyYAML yüklü değil, settings.yaml okunamadı.")
        return

    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:
        rprint(f"[yellow]Uyarı:[/yellow] settings.yaml okunamadı: {e}")
        return

    models = data.get("models") or {}
    qwen_name = models.get("query")
    if isinstance(qwen_name, str) and qwen_name.strip():
        cfg.qwen_model = qwen_name.strip()
        rprint(f"[cyan]Ayar:[/cyan] Qwen modeli -> {cfg.qwen_model}")


def _default_qa_path(cmd_arg: Optional[str]) -> Path:
    """
    Kullanıcı path vermezse varsayılanı kullan.
    Varsayılan: data/query_data/qa10_kvkk.json
    """
    base = Path("data/query_data")
    if cmd_arg is None:
        return base / "qa10_kvkk.json"

    p = Path(cmd_arg)
    if p.is_absolute():
        return p
    if str(p).startswith(str(base)):
        return p
    return base / p


def _choose_output_paths(
    qa_path: Path,
    out_json: Optional[str],
    out_xlsx: Optional[str],
) -> tuple[Path, Path]:
    """
    Çıktılar soru dosyasının bulunduğu klasöre yazılır.
    Kullanıcı path vermezse isimler 'run.json' ve 'run.xlsx' olur.
    """
    parent = qa_path.parent
    json_path = Path(out_json) if out_json else (parent / "run.json")
    xlsx_path = Path(out_xlsx) if out_xlsx else (parent / "run.xlsx")

    if not json_path.is_absolute():
        json_path = parent / json_path.name
    if not xlsx_path.is_absolute():
        xlsx_path = parent / xlsx_path.name

    return json_path, xlsx_path


def _row_from_output(idx: int, out: OutputSchema) -> Dict[str, Any]:
    """
    OutputSchema -> düz dict (JSON ve Excel için).
    Pydantic v2: .model_dump(), v1: .dict()
    """
    try:
        data = out.model_dump()  # pydantic v2
    except Exception:
        data = out.dict()  # type: ignore

    ref = data.get("reference") or {}
    return {
        "idx": idx,
        "query": data.get("query", ""),
        "answer": data.get("answer", ""),
        "doc_id": ref.get("doc_id", ""),
        "page": ref.get("page", None),
    }


# -------------------- Komutlar --------------------

def cmd_build(pdf_dir: Optional[str], prompt_yaml: str, settings_path: Optional[str]):
    """
    PDF indeksleme. Varsayılan klasör: data/query_data
    """
    cfg = Settings()
    _load_models_from_settings(settings_path, cfg)

    agent = QueryAgent(cfg, prompt_yaml)
    target_dir = pdf_dir or "data/query_data"
    agent.build_index(target_dir)
    rprint(f"[green]✅ İndeks oluşturuldu.[/green]  [dim]Klasör:[/dim] {target_dir}")


def cmd_ask(question: str, k: int, prompt_yaml: str, settings_path: Optional[str]):
    cfg = Settings()
    _load_models_from_settings(settings_path, cfg)

    agent = QueryAgent(cfg, prompt_yaml)
    out = agent.ask(question, k=k)

    if isinstance(out, OutputSchema):
        row = _row_from_output(1, out)
        rprint(json.dumps(row, ensure_ascii=False, indent=2))
    else:
        rprint(out)


def cmd_batch(
    qa_json: Optional[str],
    out_json: Optional[str],
    out_xlsx: Optional[str],
    k: int,
    prompt_yaml: str,
    settings_path: Optional[str],
):
    cfg = Settings()
    _load_models_from_settings(settings_path, cfg)

    agent = QueryAgent(cfg, prompt_yaml)

    qa_path = _default_qa_path(qa_json)
    rprint(f"Soru dosyası: {qa_path}")

    items = json.loads(qa_path.read_text(encoding="utf-8"))

    rows: List[Dict[str, Any]] = []
    for i, it in enumerate(items, 1):
        # Soru ve expected'i oku
        if isinstance(it, str):
            q = it
            expected = ""
        else:
            q = it.get("query", "")
            expected = (it.get("expected") or "")

        pred = agent.ask(q, k=k)

        # OutputSchema ise referansı al, expected'i ekle
        if isinstance(pred, OutputSchema):
            row = _row_from_output(i, pred)
            row["expected"] = expected
        else:
            row = {
                "idx": i,
                "query": q,
                "expected": expected,  # <-- İSTENEN KOLON
                "answer": (pred or "").strip(),
                "doc_id": "",
                "page": None,
            }

        rows.append(row)

        rprint(f"[white]{i:02d}[/white] [bold]Q:[/bold] {q}")
        rprint(
            f"    [dim]Expected:[/dim] {expected}  "
            f"[dim]Ans:[/dim] {row['answer']}  "
            f"[dim]Ref:[/dim] {row.get('doc_id','')} s.{row.get('page')}"
        )

    # Çıktı dosya yollarını seç (aynı klasöre yaz)
    out_json_path, out_xlsx_path = _choose_output_paths(qa_path, out_json, out_xlsx)

    # JSON yaz (expected dahil)
    out_json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    rprint(f"[green]JSON kaydedildi →[/green] {out_json_path}")

    # Excel yaz — kolon sırası: idx, query, expected, answer, doc_id, page
    df = pd.DataFrame(rows, columns=["idx", "query", "expected", "answer", "doc_id", "page"])
    df.to_excel(out_xlsx_path, index=False)
    rprint(f"[green]Excel kaydedildi →[/green] {out_xlsx_path}")

    rprint(f"[bold]Bitti.[/bold] Toplam: {len(rows)}")


# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # build
    b = sub.add_parser("build", help="PDF klasörünü indeksle")
    b.add_argument("--pdf_dir", default=None, help="PDF klasörü (varsayılan: data/query_data)")
    b.add_argument("--prompts", default="prompts/query_prompt.yaml", help="Prompt YAML yolu")
    b.add_argument("--settings", default="config/settings.yaml", help="Ayar dosyası (yaml)")

    # ask
    a = sub.add_parser("ask", help="Tek soru sor")
    a.add_argument("--q", required=True, help="Soru metni")
    a.add_argument("--k", type=int, default=5, help="Kaç belge getirilsin (top_k)")
    a.add_argument("--prompts", default="prompts/query_prompt.yaml", help="Prompt YAML yolu")
    a.add_argument("--settings", default="config/settings.yaml", help="Ayar dosyası (yaml)")

    # batch
    bt = sub.add_parser("batch", help="JSON soru seti çalıştır")
    bt.add_argument("--qa_json", default=None, help="Soru listesi JSON (varsayılan: data/query_data/qa10_kvkk.json)")
    bt.add_argument("--out_json", default=None, help="Çıktı JSON (varsayılan: <qa_json klasörü>/run.json)")
    bt.add_argument("--out_xlsx", default=None, help="Çıktı Excel (varsayılan: <qa_json klasörü>/run.xlsx)")
    bt.add_argument("--k", type=int, default=5, help="Kaç belge getirilsin (top_k)")
    bt.add_argument("--prompts", default="prompts/query_prompt.yaml", help="Prompt YAML yolu")
    bt.add_argument("--settings", default="config/settings.yaml", help="Ayar dosyası (yaml)")

    args = ap.parse_args()

    if args.cmd == "build":
        return cmd_build(args.pdf_dir, args.prompts, args.settings)
    if args.cmd == "ask":
        return cmd_ask(args.q, args.k, args.prompts, args.settings)
    if args.cmd == "batch":
        return cmd_batch(args.qa_json, args.out_json, args.out_xlsx, args.k, args.prompts, args.settings)


if __name__ == "__main__":
    main()
