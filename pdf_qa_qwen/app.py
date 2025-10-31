import argparse
import json
from pathlib import Path
from rich.prompt import Prompt
from rich import print as rprint
import pandas as pd

from src.engine import QAPipeline
from src.utils import unwrap_pred, normalize_for_compare

def run_interactive(pdf_path: str | None, top_k: int, prompt_yaml: str, prompt_key: str):
    qa = QAPipeline(pdf_path=pdf_path, prompt_yaml_path=prompt_yaml, prompt_key=prompt_key)
    rprint("[bold cyan]PDF QA (Qwen3-8B, Extractive)[/bold cyan]  Çıkış için boş Enter.")
    while True:
        q = Prompt.ask("Soru").strip()
        if not q:
            break
        rprint(qa.ask(q, top_k=top_k))

def run_build(pdf_path: str, prompt_yaml: str, prompt_key: str):
    qa = QAPipeline(pdf_path=pdf_path, prompt_yaml_path=prompt_yaml, prompt_key=prompt_key)
    qa.build_index()
    rprint("[green]✅ İndeks oluşturuldu.[/green]")

def run_batch(qa_json: str, pdf_path: str | None, out_json: str | None, out_xlsx: str | None, top_k: int, prompt_yaml: str, prompt_key: str):
    qa = QAPipeline(pdf_path=pdf_path, prompt_yaml_path=prompt_yaml, prompt_key=prompt_key)

    with open(qa_json, "r", encoding="utf-8") as f:
        items = json.load(f)

    rows = []
    n = len(items)
    rprint(f"[cyan]Toplam {n} soru işlenecek...[/cyan]")

    for idx, it in enumerate(items, 1):
        q = it.get("query", "").strip()
        expected = (it.get("expected", "") or "").strip()
        answerable = bool(it.get("answerable", True))

        pred = qa.ask(q, top_k=top_k)                 # '<<<...>>>' ya da 'BELİRTİLMEMİŞ'
        pred_unwrapped = unwrap_pred(pred)

        if not answerable:
            is_correct = (pred_unwrapped == "BELİRTİLMEMİŞ")
        else:
            is_correct = (
                pred_unwrapped != "BELİRTİLMEMİŞ"
                and normalize_for_compare(pred_unwrapped) == normalize_for_compare(expected)
            )

        rows.append({
            "idx": idx,
            "query": q,
            "expected": expected,
            "answerable": answerable,
            "predicted": pred,
            "pred_unwrapped": pred_unwrapped,
            "is_correct": bool(is_correct),
        })

        tag = "✅" if is_correct else "❌"
        rprint(f"[white]{idx:02d}[/white] {tag}  [bold]Q:[/bold] {q}")
        rprint(f"    [dim]Pred:[/dim] {pred}")

    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        rprint(f"[green]JSON kaydedildi →[/green] {out_json}")

    if out_xlsx:
        df = pd.DataFrame(rows, columns=["idx","query","expected","answerable","predicted","pred_unwrapped","is_correct"])
        Path(out_xlsx).parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(out_xlsx, index=False)
        rprint(f"[green]Excel kaydedildi →[/green] {out_xlsx}")

    total = len(rows)
    correct = sum(1 for r in rows if r["is_correct"])
    rprint(f"[bold]Bitti.[/bold] Doğru: {correct}/{total}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=str, help="PDF yolu (ilk kurulumda veya değiştiğinde verin)")
    ap.add_argument("--build", action="store_true", help="PDF'ten indeks oluştur")
    ap.add_argument("--ask", type=str, help="Tek bir soru sor ve çık")
    ap.add_argument("--topk", type=int, default=3, help="Kaç sayfayı sorgula (varsayılan 3)")

    # JSON batch modu
    ap.add_argument("--qa_json", type=str, help="Soru listesi JSON dosyası")
    ap.add_argument("--out_json", type=str, help="Çıktı JSON dosya yolu")
    ap.add_argument("--out_xlsx", type=str, help="Çıktı Excel dosya yolu")

    # Prompt YAML
    ap.add_argument("--prompt_yaml", type=str, default="prompts/prompts.yaml", help="Prompt YAML dosya yolu")
    ap.add_argument("--prompt_key", type=str, default="extractive_tr.system", help="Prompt dot-key (ör. extractive_tr.system)")

    args = ap.parse_args()

    if args.build:
        if not args.pdf:
            raise SystemExit("--build için --pdf zorunlu")
        return run_build(args.pdf, args.prompt_yaml, args.prompt_key)

    if args.qa_json:
        return run_batch(
            qa_json=args.qa_json,
            pdf_path=args.pdf,
            out_json=args.out_json,
            out_xlsx=args.out_xlsx,
            top_k=args.topk,
            prompt_yaml=args.prompt_yaml,
            prompt_key=args.prompt_key,
        )

    if args.ask:
        qa = QAPipeline(pdf_path=args.pdf, prompt_yaml_path=args.prompt_yaml, prompt_key=args.prompt_key)
        r = qa.ask(args.ask, top_k=args.topk)
        rprint(r)
        return

    return run_interactive(pdf_path=args.pdf, top_k=args.topk, prompt_yaml=args.prompt_yaml, prompt_key=args.prompt_key)

if __name__ == "__main__":
    main()
