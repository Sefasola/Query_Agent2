import argparse
import json
from pathlib import Path
from rich import print as rprint
import pandas as pd

from src.config import Settings
from src.qa_agent import QueryAgent

def cmd_build(pdf_dir: str, prompt_yaml: str):
    cfg = Settings()
    agent = QueryAgent(cfg, prompt_yaml)
    agent.build_index(pdf_dir)
    rprint("[green]✅ İndeks oluşturuldu.[/green]")

def cmd_ask(question: str, k: int, prompt_yaml: str):
    cfg = Settings()
    agent = QueryAgent(cfg, prompt_yaml)
    ans = agent.ask(question, k=k)
    rprint(ans)

def cmd_batch(qa_json: str, out_json: str | None, out_xlsx: str | None, k: int, prompt_yaml: str):
    cfg = Settings()
    agent = QueryAgent(cfg, prompt_yaml)

    items = json.loads(Path(qa_json).read_text(encoding="utf-8"))
    rows = []
    for i, it in enumerate(items, 1):
        q = it["query"]
        expected = (it.get("expected") or "").strip()
        answerable = bool(it.get("answerable", True))
        pred = agent.ask(q, k=k)

        # Değerlendirme:
        if not answerable:
            is_correct = (pred == "BELİRTİLMEMİŞ")
        else:
            # Basit eşleşme (noktalama farkına hassas; dilersen normalize ederek kıyasla)
            is_correct = (pred.strip().rstrip(" .;:") == expected.strip().rstrip(" .;:"))

        rows.append({
            "idx": i,
            "query": q,
            "expected": expected,
            "answerable": answerable,
            "predicted": pred,
            "is_correct": bool(is_correct),
        })

        tag = "✅" if is_correct else "❌"
        rprint(f"[white]{i:02d}[/white] {tag} [bold]Q:[/bold] {q}")
        rprint(f"    [dim]Pred:[/dim] {pred}")

    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        rprint(f"[green]JSON kaydedildi →[/green] {out_json}")

    if out_xlsx:
        df = pd.DataFrame(rows, columns=["idx","query","expected","answerable","predicted","is_correct"])
        Path(out_xlsx).parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(out_xlsx, index=False)
        rprint(f"[green]Excel kaydedildi →[/green] {out_xlsx}")

    total = len(rows)
    correct = sum(1 for r in rows if r["is_correct"])
    rprint(f"[bold]Bitti.[/bold] Doğru: {correct}/{total}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="PDF klasörünü indeksle")
    b.add_argument("--pdf_dir", required=True, help="PDF klasörü (ör. data/pdfs)")
    b.add_argument("--prompts", default="prompts/prompts.yaml", help="Prompt YAML yolu")

    a = sub.add_parser("ask", help="Soru sor")
    a.add_argument("--q", required=True, help="Soru metni")
    a.add_argument("--k", type=int, default=5, help="Kaç belge getirilsin (top_k)")
    a.add_argument("--prompts", default="prompts/prompts.yaml", help="Prompt YAML yolu")

    bt = sub.add_parser("batch", help="JSON soru seti çalıştır")
    bt.add_argument("--qa_json", required=True, help="Soru listesi JSON")
    bt.add_argument("--out_json", help="Çıktı JSON")
    bt.add_argument("--out_xlsx", help="Çıktı Excel")
    bt.add_argument("--k", type=int, default=5, help="Kaç belge getirilsin (top_k)")
    bt.add_argument("--prompts", default="prompts/prompts.yaml", help="Prompt YAML yolu")

    args = ap.parse_args()

    if args.cmd == "build":
        return cmd_build(args.pdf_dir, args.prompts)
    if args.cmd == "ask":
        return cmd_ask(args.q, args.k, args.prompts)
    if args.cmd == "batch":
        return cmd_batch(args.qa_json, args.out_json, args.out_xlsx, args.k, args.prompts)

if __name__ == "__main__":
    main()
