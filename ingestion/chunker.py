from pydantic import BaseModel
from ingestion.loader import PageText


class Chunk(BaseModel):
    text: str
    page_number: int  # page the chunk starts on


def _extend_to_word_boundary(text: str, end: int) -> int:
    """Push `end` forward to the next whitespace so a chunk never splits a word."""
    while end < len(text) and not text[end].isspace():
        end += 1
    return end


def chunk_pages(pages: list[PageText], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Chunk]:
    """Slide a word-boundary-respecting window over the document's full text, tracking
    which page each chunk starts on. Chunks may span a page break — a fixed-size,
    per-page chunker would produce degenerate tiny chunks for short pages."""
    if chunk_overlap >= chunk_size:
        raise ValueError('chunk_overlap must be smaller than chunk_size')

    full_text = ''
    page_starts: list[tuple[int, int]] = []  # (offset into full_text, page_number)
    for page in pages:
        page_starts.append((len(full_text), page.page_number))
        full_text += page.text
        if not full_text.endswith('\n'):
            full_text += '\n'

    full_text = full_text.strip()
    if not full_text:
        return []

    def page_for_offset(offset: int) -> int:
        page_number = page_starts[0][1]
        for start, number in page_starts:
            if start > offset:
                break
            page_number = number
        return page_number

    chunks = []
    start = 0
    while start < len(full_text):
        end = _extend_to_word_boundary(full_text, min(start + chunk_size, len(full_text)))
        text = full_text[start:end].strip()

        if text:
            chunks.append(Chunk(text=text, page_number=page_for_offset(start)))

        if end >= len(full_text):
            break

        # Word-boundary-align the next start too — end - chunk_overlap is an arbitrary
        # character offset that usually lands mid-word without this.
        start = _extend_to_word_boundary(full_text, max(end - chunk_overlap, start + 1))

    return chunks
