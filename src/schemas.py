from __future__ import annotations
from pydantic import BaseModel

class InputSchema(BaseModel):
    query: str       # Soru metni
    pdf_path: str    # PDF tam yolu (opsiyonel kullanım için alan mevcut)

class Reference(BaseModel):
    doc_id: str      # PDF dosya adı
    page: int        # 1-tabanlı sayfa numarası

class OutputSchema(BaseModel):
    # ÇIKTI SIRASI: query -> answer -> reference
    query: str
    answer: str
    reference: Reference
