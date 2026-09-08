import argparse
import time
from pathlib import Path
from ingestion.loader import load_pdf
from ingestion.chunker import chunk_pages
from ingestion.embedder import embed_texts
from ingestion.store import get_client, ensure_collection, upsert_chunks
from ingestion.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

# embed_texts() already retries individual embedding calls internally (see
# EMBEDDING_TPM_LIMIT pacing + 429 backoff in embedder.py); this is an outer safety net
# for when a whole PDF still fails — a rate limit that outlasts embed_texts' own retry
# budget, a dropped connection during upsert, etc. — so one bad file doesn't abort a
# large batch. A file that's still failing after this many attempts (e.g. a genuinely
# corrupt PDF) is skipped rather than retried forever.
_MAX_FILE_RETRIES = 3
_FILE_RETRY_DELAY_SECONDS = 60


def ingest_pdf(
    pdf_path: str,
    collection_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    client=None,
    verbose: bool = True,
) -> int:
    """Load a PDF, chunk its text, embed each chunk, and upsert into a Qdrant collection.
    Returns the number of points upserted. Pass `client` to reuse a connection across
    multiple calls (see ingest_directory) instead of opening a new one per file."""
    source = Path(pdf_path).name

    pages = load_pdf(pdf_path)
    if verbose:
        print(f'Loaded {len(pages)} page(s) from {source}')

    chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if verbose:
        print(f'Split into {len(chunks)} chunk(s)')

    if not chunks:
        return 0

    vectors = embed_texts([chunk.text for chunk in chunks])

    client = client or get_client()
    ensure_collection(client, collection_name, vector_size=len(vectors[0]))
    count = upsert_chunks(client, collection_name, chunks, vectors, source=source)

    if verbose:
        print(f'Upserted {count} point(s) into collection "{collection_name}"')

    return count


def _ingest_pdf_with_retry(
    pdf_path: Path,
    collection_name: str,
    chunk_size: int,
    chunk_overlap: int,
    client,
    verbose: bool,
) -> int:
    for attempt in range(_MAX_FILE_RETRIES):
        try:
            return ingest_pdf(
                str(pdf_path), collection_name,
                chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                client=client, verbose=verbose,
            )
        except Exception as e:
            if attempt == _MAX_FILE_RETRIES - 1:
                raise
            if verbose:
                print(
                    f'  {pdf_path.name} failed ({e}) — waiting {_FILE_RETRY_DELAY_SECONDS}s '
                    f'and retrying (attempt {attempt + 2}/{_MAX_FILE_RETRIES})'
                )
            time.sleep(_FILE_RETRY_DELAY_SECONDS)


def ingest_directory(
    directory: str,
    collection_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    recursive: bool = False,
    verbose: bool = True,
) -> dict[str, int]:
    """Ingest every PDF in a directory into the same collection. A file that keeps failing
    (rate limit, corrupt PDF, etc.) is retried a few times, then skipped — one bad file
    doesn't abort the rest of the batch. Returns {filename: points_upserted} for files
    that succeeded."""
    directory = Path(directory)
    pattern = '**/*.pdf' if recursive else '*.pdf'
    pdf_paths = sorted(directory.glob(pattern))

    if not pdf_paths:
        if verbose:
            print(f'No PDF files found in {directory}')
        return {}

    client = get_client()
    results: dict[str, int] = {}
    failures: dict[str, str] = {}

    for i, pdf_path in enumerate(pdf_paths, start=1):
        if verbose:
            print(f'[{i}/{len(pdf_paths)}] {pdf_path.name}')
        try:
            results[pdf_path.name] = _ingest_pdf_with_retry(
                pdf_path, collection_name, chunk_size, chunk_overlap, client, verbose,
            )
        except Exception as e:
            failures[pdf_path.name] = str(e)
            if verbose:
                print(f'  GIVING UP on {pdf_path.name} after {_MAX_FILE_RETRIES} attempts: {e}')

    if verbose:
        total_points = sum(results.values())
        print(f'\nDone: {len(results)} succeeded ({total_points} point(s) total), {len(failures)} failed.')
        if failures:
            print('Failed files:')
            for name, err in failures.items():
                print(f'  {name}: {err}')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ingest a PDF file or a directory of PDFs into a Qdrant collection.')
    parser.add_argument('path', help='Path to a PDF file or a directory of PDF files to ingest')
    parser.add_argument('--collection', required=True, help='Qdrant collection name')
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument('--chunk-overlap', type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument('--recursive', action='store_true', help='Recurse into subdirectories when path is a directory')
    args = parser.parse_args()

    target = Path(args.path)
    if target.is_dir():
        ingest_directory(
            target, args.collection,
            chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
            recursive=args.recursive,
        )
    elif target.is_file():
        ingest_pdf(str(target), args.collection, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    else:
        parser.error(f'{target} is not a valid file or directory')
