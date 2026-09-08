from pathlib import Path
from pydantic import BaseModel
from pypdf import PdfReader


class PageText(BaseModel):
    page_number: int  # 1-indexed
    text: str


def load_pdf(path: str | Path) -> list[PageText]:
    """Extract text from a PDF, one entry per page (in order, including blank pages)."""
    reader = PdfReader(str(path))
    return [
        PageText(page_number=i, text=page.extract_text() or '')
        for i, page in enumerate(reader.pages, start=1)
    ]
