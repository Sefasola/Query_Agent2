# app.py
from __future__ import annotations
import argparse
import json
from pathlib import Path
from rich import print as rprint
import pandas as pd

from src.config import Settings
from src.qa_agent import QueryAgent

# Varsayılan veri klasörü (PDF + Soru JSON + Çıktılar)
DEFAULT_DATA_DIR = Path("data/query_data")
DEFAULT_PROMPTS = "prompts/prompts.yaml"
DEFAULT_OUT_JSON_NAME = "run.json"
DEFAULT_OUT_XLSX_NAME = "run.xlsx"
PREFER_QA_FILE = "qa10_kvkk.json"


def autodetect_qa_json(data_dir: Path) -> Path:
    """
    Önce data_dir/PREFER_QA_FILE varsa onu al.
    Yoksa klasördeki .json dosyalarını tara:
      - 'run*.json' gibi çıktı dosyalarını hariç tut.
      - Bulduğu ilk JSON'u seç (alfabetik).
    Hiçbiri yoksa anlaşılır bir hata ver.
    """
    cand = data_dir / PREFER_QA_FILE
    if cand.exists():
        return cand

    jsons = sorted(
        p for p in data_dir.glob("*.json")
        if p.name.lower()[:3] != "run"  # run.json gibi çıktı dosyalarını at
    )
    if not jsons:
        raise FileNotFoundError(
            f"'{data_dir}' içinde giriş için kullanılacak herhangi bir .json bulunamadı. "
            f"Lütfen bir soru dosyası ekleyin (örn. {PREFER_QA_FILE})."
        )
    return jsons[0]


def resolve_out_path(arg: str | None, base_dir: Path, default_name: str) -> Path:
    """
    Çıkış yolu:
      - None ise: base_dir/default_name
      - Mutlaksa: olduğu gibi
      - Göreceliyse: base_dir/arg
    """
    if not arg:
        return base_dir / default_name
    p = Path(arg)
    if p.is_absolute():
        return p
    return base_dir / p


def cmd_build(pdf_dir: str | None, prompt_yaml: str | None):
    cfg = Settings()
    agent = QueryAgent(cfg, prompt_yaml or DEFAULT_PROMPTS)

    # Yol verilmediyse varsayılan klasör
    pdf_root = Path(pdf_dir) if pdf_dir else DEFAULT_DATA_DIR
    agent.build_index(str(pdf_root))
    rprint(f"[green]✅ İndeks oluşturuldu.[/green] [dim]Kaynak klasör:[/dim] {pdf_root}")


def cmd_ask(question: str, k: int, prompt_yaml: str | None):
    cfg = Settings()
    agent = QueryAgent(cfg, prompt_yaml or DEFAULT_PROMPTS)
    ans = agent.ask(question, k=k)
    rprint(ans)


def cmd_batch(
    qa_json: str | None,
    out_json: str | None,
    out_xlsx: str | None,
    k: int,
    prompt_yaml: str | None,
    settings_yaml: str | None,
):
    cfg = Settings()
    agent = QueryAgent(cfg, prompt_yaml or DEFAULT_PROMPTS)

    # Soru dosyası otomatik tespit (yol verilmemişse)
    base_dir = DEFAULT_DATA_DIR
    qa_path = Path(qa_json) if qa_json else autodetect_qa_json(base_dir)
    if not qa_path.is_absolute():
        qa_path = base_dir / qa_path
    rprint(f"[dim]Soru dosyası:[/dim] {qa_path}")

    # Çıktılar her zaman soru dosyasının klasörüne
    base_dir = qa_path.parent
    out_json_path = resolve_out_path(out_json, base_dir, DEFAULT_OUT_JSON_NAME)
    out_xlsx_path = resolve_out_path(out_xlsx, base_dir, DEFAULT_OUT_XLSX_NAME)
    base_dir.mkdir(parents=True, exist_ok=True)

    items = json.loads(qa_path.read_text(encoding="utf-8"))

    rows = []
    for i, it in enumerate(items, 1):
        q = it["query"]
        pred = agent.ask(q, k=k)

        rows.append({
            "idx": i,
            "query": q,
            "predicted": pred,
        })

        rprint(f"[white]{i:02d}[/white] [bold]Q:[/bold] {q}")
        rprint(f"    [dim]Pred:[/dim] {pred}")

    # JSON
    out_json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    rprint(f"[green]JSON kaydedildi →[/green] {out_json_path}")

    # Excel
    df = pd.DataFrame(rows, columns=["idx", "query", "predicted"])
    df.to_excel(out_xlsx_path, index=False)
    rprint(f"[green]Excel kaydedildi →[/green] {out_xlsx_path}")

    rprint(f"[bold]Bitti.[/bold] Toplam: {len(rows)} kayıt")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # build: artık tamamsıyla opsiyonel
    b = sub.add_parser("build", help="PDF klasörünü indeksle (varsayılan: data/query_data)")
    b.add_argument("--pdf_dir", help="PDF klasörü (opsiyonel; verilmezse data/query_data)")
    b.add_argument("--prompts", help="Prompt YAML yolu (opsiyonel; varsayılan: prompts/prompts.yaml)")

    a = sub.add_parser("ask", help="Soru sor")
    a.add_argument("--q", required=True, help="Soru metni")
    a.add_argument("--k", type=int, default=5, help="Kaç belge getirilsin (top_k)")
    a.add_argument("--prompts", help="Prompt YAML yolu (opsiyonel)")

    bt = sub.add_parser("batch", help="JSON soru seti çalıştır")
    bt.add_argument("--qa_json", help="Soru listesi JSON (opsiyonel; verilmezse otomatik bulunur)")
    bt.add_argument("--out_json", help="Çıktı JSON (opsiyonel; verilmezse run.json)")
    bt.add_argument("--out_xlsx", help="Çıktı Excel (opsiyonel; verilmezse run.xlsx)")
    bt.add_argument("--k", type=int, default=5, help="Kaç belge getirilsin (top_k)")
    bt.add_argument("--prompts", help="Prompt YAML yolu (opsiyonel)")
    bt.add_argument("--settings", help="Model ayarları (opsiyonel)")

    args = ap.parse_args()

    if args.cmd == "build":
        return cmd_build(args.pdf_dir, args.prompts)
    if args.cmd == "ask":
        return cmd_ask(args.q, args.k, args.prompts)
    if args.cmd == "batch":
        return cmd_batch(args.qa_json, args.out_json, args.out_xlsx, args.k, args.prompts, args.settings)


if __name__ == "__main__":
    main()
