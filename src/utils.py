import re

def clean_text(s: str) -> str:
    # Basit normalizasyon: boşlukları sadeleştir, kontrol karakterlerini sil
    s = s.replace("\x00", " ").replace("\u200b", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

_ANS_RE = re.compile(r"^<<<(.*?)>>>$", re.DOTALL)

def parse_extractive_answer(model_text: str) -> str | None:
    """
    Model çıktısı biçiminden tek satır cevabı ayıkla.
    Biçim: <<<...>>>  değilse None döner.
    """
    m = _ANS_RE.match(model_text.strip())
    if not m:
        # Tam eşleşmediyse tek satıra indirip tekrar dene
        one_line = " ".join(model_text.split())
        m = _ANS_RE.match(one_line) if one_line else None
        if not m:
            return None
    return m.group(1).strip()

def is_verbatim_substring(needle: str, haystack: str) -> bool:
    """Cevap tam olarak sayfa metni içinde tek parça olarak geçiyor mu? (birebir)"""
    return needle in haystack

# ---- Batch değerlendirme yardımcıları ----

def unwrap_pred(pred: str) -> str:
    """
    '<<<...>>>' sarmalamasını kaldırır; BELİRTİLMEMİŞ'i olduğu gibi döner.
    """
    t = pred.strip()
    if t == "BELİRTİLMEMİŞ":
        return t
    m = _ANS_RE.match(t)
    return m.group(1).strip() if m else t

def normalize_for_compare(s: str) -> str:
    """
    Karşılaştırma için basit normalize:
      - kenar boşlukları
      - sondaki . ; : işaretlerini kırp
      - birden çok boşluğu teke indir
    """
    s = (s or "").strip()
    s = s.rstrip(" .;:")  # sondaki noktalama toleransı
    s = " ".join(s.split())
    return s
