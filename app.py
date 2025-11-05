import argparse
import json
from pathlib import Path
from rich import print as rprint
import pandas as pd

from src.config import Settings
from src.qa_agent import QueryAgent
from src.schemas import InputSchema  # şema entegrasyonu

# Tüm I/O için varsayılan kök klasör
DEFAULT_DATA_DIR = Path("data/query_data")


def resolve_in_data_dir(path_like: str | None, default_name: str | None = None) -> Path:
    """
    - path_like abs. path ise doğrudan kullan.
    - relatif ise ve mevcutsa onu kullan.
    - aksi halde DEFAULT_DATA_DIR altına yerleştir.
    - None ise default_name zorunludur -> DEFAULT_DATA_DIR/default_name
    """
    if path_like:
        p = Path(path_like)
        if p.is_absolute():
            return p
        if p.exists():
            return p
        return DEFAULT_DATA_DIR / p
    assert default_name is not None, "default_name gerekiyor"
    return DEFAULT_DATA_DIR / default_name


def cmd_build(pdf_dir: str, prompt_yaml: str, settings_yaml: str | None):
    cfg = Settings.from_yaml(settings_yaml)
    agent = QueryAgent(cfg, prompt_yaml)

    # pdf_dir verilmişse kullan; değilse DEFAULT_DATA_DIR
    pdf_root = Path(pdf_dir) if pdf_dir else DEFAULT_DATA_DIR
    agent.build_index(str(pdf_root))
    rprint(f"[green]✅ İndeks oluşturuldu.[/green]  [dim]Klasör:[/dim] {pdf_root}")


def cmd_ask(question: str, k: int, prompt_yaml: str, settings_yaml: str | None):
    cfg = Settings.from_yaml(settings_yaml)
    agent = QueryAgent(cfg, prompt_yaml)

    out = agent.ask(question, k=k)  # OutputSchema döner
    rprint(f"[bold]Answer:[/bold] {out.answer}")
    rprint(f"[dim]Ref:[/dim] {out.reference.doc_id} s.{out.reference.page}")


def cmd_batch(
    qa_json: str,
    out_json: str | None,
    out_xlsx: str | None,
    k: int,
    prompt_yaml: str,
    settings_yaml: str | None,
):
    cfg = Settings.from_yaml(settings_yaml)
    agent = QueryAgent(cfg, prompt_yaml)

    # Soru dosyasını data/query_data altında çöz
    qa_path = resolve_in_data_dir(qa_json, default_name="qa10_kvkk.json")
    rprint(f"[cyan]Soru dosyası:[/cyan] {qa_path}")

    items = json.loads(qa_path.read_text(encoding="utf-8"))

    rows = []
    correct_count = 0
    total = 0

    for i, it in enumerate(items, 1):
        q = it["query"]
        expected = (it.get("expected") or "").strip()
        answerable = bool(it.get("answerable", True))

        out = agent.ask(q, k=k)  # OutputSchema
        pred = out.answer

        # Dahili değerlendirme (Excel'e yazmıyoruz)
        if not answerable:
            is_correct = (pred == "BELİRTİLMEMİŞ")
        else:
            is_correct = (pred.strip().rstrip(" .;:") == expected.strip().rstrip(" .;:"))

        total += 1
        if is_correct:
            correct_count += 1

        rows.append({
            "idx": i,
            "query": out.query,
            "expected": expected,
            "predicted": pred,
            "ref_doc": out.reference.doc_id,
            "ref_page": out.reference.page,
        })

        tag = "✅" if is_correct else "❌"
        rprint(f"[white]{i:02d}[/white] {tag} [bold]Q:[/bold] {out.query}")
        rprint(f"    [dim]Pred:[/dim] {pred}  [dim]Ref:[/dim] {out.reference.doc_id} s.{out.reference.page}")

    # Çıkış dosyaları verilmediyse data/query_data altına otomatik ata
    if out_json is None:
        out_json = (DEFAULT_DATA_DIR / f"{qa_path.stem}_pred.json").as_posix()
    if out_xlsx is None:
        out_xlsx = (DEFAULT_DATA_DIR / f"{qa_path.stem}_pred.xlsx").as_posix()

    out_json_path = resolve_in_data_dir(out_json)
    out_xlsx_path = resolve_in_data_dir(out_xlsx)

    # JSON kaydet
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    rprint(f"[green]JSON kaydedildi →[/green] {out_json_path}")

    # Excel kaydet (answerable/is_correct sütunları YOK)
    df = pd.DataFrame(rows, columns=["idx", "query", "expected", "predicted", "ref_doc", "ref_page"])
    out_xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_xlsx_path, index=False)
    rprint(f"[green]Excel kaydedildi →[/green] {out_xlsx_path}")

    rprint(f"[bold]Bitti.[/bold] Doğru: {correct_count}/{total}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    default_yaml = "config/settings.yaml"

    # build: varsayılan pdf kökü data/query_data
    b = sub.add_parser("build", help="PDF klasörünü indeksle")
    b.add_argument("--pdf_dir", default="data/query_data",
                   help="PDF klasörü (varsayılan: data/query_data)")
    b.add_argument("--prompts", default="prompts/prompts.yaml", help="Prompt YAML yolu")
    b.add_argument("--settings", default=default_yaml, help="config/settings.yaml yolu (opsiyonel)")

    a = sub.add_parser("ask", help="Soru sor")
    a.add_argument("--q", required=True, help="Soru metni")
    a.add_argument("--k", type=int, default=5, help="Kaç belge getirilsin (top_k)")
    a.add_argument("--prompts", default="prompts/prompts.yaml", help="Prompt YAML yolu")
    a.add_argument("--settings", default=default_yaml, help="config/settings.yaml yolu (opsiyonel)")

    bt = sub.add_parser("batch", help="JSON soru seti çalıştır")
    # Burada default dosya adını veriyoruz; path çözümlemesini data/query_data altında yapacağız
    bt.add_argument("--qa_json", default="qa10_kvkk.json",
                    help="Soru listesi JSON (varsayılan: data/query_data/qa10_kvkk.json)")
    bt.add_argument("--out_json", help="Çıktı JSON (verilmezse data/query_data altına otomatik yazılır)")
    bt.add_argument("--out_xlsx", help="Çıktı Excel (verilmezse data/query_data altına otomatik yazılır)")
    bt.add_argument("--k", type=int, default=5, help="Kaç belge getirilsin (top_k)")
    bt.add_argument("--prompts", default="prompts/prompts.yaml", help="Prompt YAML yolu")
    bt.add_argument("--settings", default=default_yaml, help="config/settings.yaml yolu (opsiyonel)")

    args = ap.parse_args()

    if args.cmd == "build":
        return cmd_build(args.pdf_dir, args.prompts, args.settings)
    if args.cmd == "ask":
        return cmd_ask(args.q, args.k, args.prompts, args.settings)
    if args.cmd == "batch":
        return cmd_batch(args.qa_json, args.out_json, args.out_xlsx, args.k, args.prompts, args.settings)


if __name__ == "__main__":
    main()
