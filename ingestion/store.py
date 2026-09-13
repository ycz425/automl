from uuid import uuid5, NAMESPACE_URL
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from ingestion.chunker import Chunk
from ingestion.config import QDRANT_ENDPOINT, QDRANT_API_KEY, QDRANT_PATH, EMBEDDING_DIMENSIONS

# Deterministic per (source, chunk index) rather than a fresh uuid4() per upsert — so
# re-upserting the same chunk (e.g. pipeline.py retrying a PDF after a later chunk
# failed) overwrites the same point instead of inserting a duplicate.
def chunk_point_id(source: str, chunk_index: int) -> str:
    return str(uuid5(NAMESPACE_URL, f'{source}::{chunk_index}'))


def get_client() -> QdrantClient:
    if QDRANT_ENDPOINT:
        return QdrantClient(url=QDRANT_ENDPOINT, api_key=QDRANT_API_KEY)
    if QDRANT_PATH:
        return QdrantClient(path=QDRANT_PATH)
    return QdrantClient(location=':memory:')


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int = EMBEDDING_DIMENSIONS) -> None:
    if client.collection_exists(collection_name):
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def existing_chunk_indices(client: QdrantClient, collection_name: str, source: str, num_chunks: int) -> set[int]:
    """Which of source's chunk_index range [0, num_chunks) are already upserted, so a
    retry after a partial failure can skip re-embedding (not just re-upserting) them —
    embedding is the expensive, rate-limited step; upserting a point that already exists
    is nearly free by comparison."""
    if num_chunks == 0 or not client.collection_exists(collection_name):
        return set()

    id_to_index = {chunk_point_id(source, i): i for i in range(num_chunks)}
    found = client.retrieve(
        collection_name=collection_name,
        ids=list(id_to_index.keys()),
        with_payload=False,
        with_vectors=False,
    )
    return {id_to_index[str(point.id)] for point in found}


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: list[Chunk],
    vectors: list[list[float]],
    source: str,
    indices: list[int],
) -> int:
    """indices[i] is chunks[i]'s position in the PDF's full chunk list — needed so point
    IDs stay stable across incremental (per-batch), resumed, and single-shot upserts of
    the same PDF, not just within one call."""
    if not (len(chunks) == len(vectors) == len(indices)):
        raise ValueError('chunks, vectors, and indices must be the same length')

    points = [
        PointStruct(
            id=chunk_point_id(source, index),
            vector=vector,
            payload={
                'source': source,
                'page_number': chunk.page_number,
                'text': chunk.text,
            },
        )
        for chunk, vector, index in zip(chunks, vectors, indices)
    ]

    client.upsert(collection_name=collection_name, points=points)
    return len(points)
