from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from ingestion.chunker import Chunk
from ingestion.config import QDRANT_ENDPOINT, QDRANT_API_KEY, QDRANT_PATH, EMBEDDING_DIMENSIONS


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


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: list[Chunk],
    vectors: list[list[float]],
    source: str,
) -> int:
    if len(chunks) != len(vectors):
        raise ValueError('chunks and vectors must be the same length')

    points = [
        PointStruct(
            id=str(uuid4()),
            vector=vector,
            payload={
                'source': source,
                'page_number': chunk.page_number,
                'text': chunk.text,
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    client.upsert(collection_name=collection_name, points=points)
    return len(points)
