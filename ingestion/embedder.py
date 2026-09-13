import time
import httpx
from google import genai
from google.genai import types, errors
from ingestion.config import GEMINI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, EMBEDDING_TPM_LIMIT, EMBEDDING_RPM_LIMIT

# Transient failures worth retrying beyond a 429: dropped connections / timeouts. Ingestion
# can now be paced over several minutes (see embed_texts), so a mid-run network blip is a
# real risk, not just a hypothetical one.
_RETRIABLE_NETWORK_ERRORS = (httpx.TimeoutException, httpx.ConnectError)

# The embeddings API caps how many texts can go in one request; chunk our own
# batches so ingesting a large PDF doesn't hit that limit.
_EMBED_BATCH_SIZE = 100

# Gemini's tokenizer isn't cheaply available here, so token counts are estimated from
# character length. 3.5 (a rough English-prose assumption) badly undercounted this
# corpus in practice — a batch estimated at ~27k tokens actually used ~60k real tokens
# against a real 30k TPM cap, triggering immediate 429s on the very first request of a
# run. These academic PDFs are denser than prose (math notation, citations, PDF-
# extraction artifacts), so tokens pack more per character than plain English. Lowered
# to bring the estimate roughly in line with that ~2.2x observed gap, with margin to
# spare rather than fitting the correction exactly to one measurement.
_CHARS_PER_TOKEN_ESTIMATE = 1.5
_TPM_SAFETY_MARGIN = 0.9

_MAX_RETRIES = 5
_RETRY_BASE_DELAY_SECONDS = 5


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN_ESTIMATE))


def _token_budget_batches(texts: list[str], token_budget: int) -> list[list[str]]:
    """Group texts into batches of at most _EMBED_BATCH_SIZE items that also stay
    under token_budget estimated tokens each — either cap can fill a batch first."""
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0

    for text in texts:
        tokens = _estimate_tokens(text)
        if current and (current_tokens + tokens > token_budget or len(current) >= _EMBED_BATCH_SIZE):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(text)
        current_tokens += tokens

    if current:
        batches.append(current)

    return batches


def _embed_batch_with_retry(client: genai.Client, batch: list[str], task_type: str) -> list[list[float]]:
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=EMBEDDING_DIMENSIONS,
                ),
            )
            return [embedding.values for embedding in response.embeddings]
        except errors.APIError as e:
            if e.code != 429 or attempt == _MAX_RETRIES:
                raise
            time.sleep(_RETRY_BASE_DELAY_SECONDS * (2 ** attempt))
        except _RETRIABLE_NETWORK_ERRORS:
            if attempt == _MAX_RETRIES:
                raise
            time.sleep(_RETRY_BASE_DELAY_SECONDS * (2 ** attempt))


def embed_texts(
    texts: list[str],
    client: genai.Client | None = None,
    task_type: str = 'RETRIEVAL_DOCUMENT',
    on_batch=None,
) -> list[list[float]]:
    """Embed texts in rate-limited batches. If on_batch(start_index, batch_texts,
    batch_vectors) is given, it's called right after each batch succeeds — before the
    next batch is even sent — so a caller can persist progress incrementally instead of
    losing everything embedded so far if a later batch fails."""
    if not texts:
        return []

    client = client or genai.Client(api_key=GEMINI_API_KEY)

    token_budget = int(EMBEDDING_TPM_LIMIT * _TPM_SAFETY_MARGIN)
    request_budget = max(1, int(EMBEDDING_RPM_LIMIT * _TPM_SAFETY_MARGIN))
    batches = _token_budget_batches(texts, token_budget)

    vectors: list[list[float]] = []
    window_start = time.monotonic()
    tokens_sent_in_window = 0
    requests_sent_in_window = 0
    start_index = 0

    for batch in batches:
        batch_tokens = sum(_estimate_tokens(t) for t in batch)

        elapsed = time.monotonic() - window_start
        if elapsed >= 60:
            window_start, tokens_sent_in_window, requests_sent_in_window = time.monotonic(), 0, 0
        elif tokens_sent_in_window + batch_tokens > token_budget or requests_sent_in_window + 1 > request_budget:
            time.sleep(60 - elapsed)
            window_start, tokens_sent_in_window, requests_sent_in_window = time.monotonic(), 0, 0

        batch_vectors = _embed_batch_with_retry(client, batch, task_type)
        if on_batch:
            on_batch(start_index, batch, batch_vectors)

        vectors.extend(batch_vectors)
        tokens_sent_in_window += batch_tokens
        requests_sent_in_window += 1
        start_index += len(batch)

    return vectors
