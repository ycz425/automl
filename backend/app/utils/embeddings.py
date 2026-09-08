import time
import httpx
from google import genai
from google.genai import types, errors

# The embeddings API caps how many texts can go in one request; chunk our own
# batches so embedding many texts at once doesn't hit that limit.
_EMBED_BATCH_SIZE = 100

# Gemini's tokenizer isn't cheaply available here, so token counts are estimated from
# character length. This deliberately overestimates (fewer chars/token than the ~4
# typical for English) so batching stays under tpm_limit even if the estimate is off,
# rather than risking a 429 by undercounting.
_CHARS_PER_TOKEN_ESTIMATE = 3.5
_TPM_SAFETY_MARGIN = 0.9

_MAX_RETRIES = 5
_RETRY_BASE_DELAY_SECONDS = 5

# Transient failures worth retrying beyond a 429: dropped connections / timeouts.
_RETRIABLE_NETWORK_ERRORS = (httpx.TimeoutException, httpx.ConnectError)


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


def _embed_batch_with_retry(
    client: genai.Client,
    batch: list[str],
    model: str,
    dimensions: int,
    task_type: str,
) -> list[list[float]]:
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = client.models.embed_content(
                model=model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=dimensions,
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
    client: genai.Client,
    model: str,
    dimensions: int,
    tpm_limit: int,
    task_type: str = 'RETRIEVAL_QUERY',
) -> list[list[float]]:
    if not texts:
        return []

    token_budget = int(tpm_limit * _TPM_SAFETY_MARGIN)
    batches = _token_budget_batches(texts, token_budget)

    vectors: list[list[float]] = []
    window_start = time.monotonic()
    tokens_sent_in_window = 0

    for batch in batches:
        batch_tokens = sum(_estimate_tokens(t) for t in batch)

        elapsed = time.monotonic() - window_start
        if elapsed >= 60:
            window_start, tokens_sent_in_window = time.monotonic(), 0
        elif tokens_sent_in_window + batch_tokens > token_budget:
            time.sleep(60 - elapsed)
            window_start, tokens_sent_in_window = time.monotonic(), 0

        vectors.extend(_embed_batch_with_retry(client, batch, model, dimensions, task_type))
        tokens_sent_in_window += batch_tokens

    return vectors
