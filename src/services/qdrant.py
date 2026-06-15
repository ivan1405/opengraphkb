import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from core.logging import get_logger

log = get_logger("qdrant_service")


def _deterministic_uuid(value: str) -> str:
    """Generate a deterministic UUID from a string for idempotent upserts."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


class QdrantService:
    """Vector store service backed by Qdrant.

    Uses a single collection with payload indexes for metadata filtering.
    Point IDs are deterministic UUIDs derived from chunk_id for idempotent
    upserts on reprocessing.
    """

    def __init__(
        self,
        url: str,
        api_key: str | None,
        collection_name: str,
        embedding_dimensions: int,
    ) -> None:
        self._url = url
        self._collection_name = collection_name
        self._embedding_dimensions = embedding_dimensions
        self._client = AsyncQdrantClient(
            url=url,
            api_key=api_key or None,
        )

    async def ensure_collection(self) -> None:
        """Create the collection and payload indexes if they don't exist."""
        collections = await self._client.get_collections()
        existing = {c.name for c in collections.collections}

        if self._collection_name not in existing:
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._embedding_dimensions,
                    distance=Distance.COSINE,
                ),
            )
            log.info(
                "qdrant_collection_created",
                collection=self._collection_name,
                dimensions=self._embedding_dimensions,
            )

            # Create payload indexes for filtered search
            for field_name, schema_type in [
                ("doc_id", PayloadSchemaType.KEYWORD),
                ("source_type", PayloadSchemaType.KEYWORD),
                ("company", PayloadSchemaType.KEYWORD),
                ("speaker", PayloadSchemaType.KEYWORD),
                ("topics", PayloadSchemaType.KEYWORD),
            ]:
                await self._client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
        else:
            log.info("qdrant_collection_exists", collection=self._collection_name)

    async def health_check(self) -> bool:
        """Return True if Qdrant is reachable."""
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False

    async def upsert_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """Batch upsert chunk vectors and payloads.

        Each chunk dict must contain:
            - chunk_id (str): Unique chunk identifier.
            - vector (list[float]): Embedding vector.
            - payload (dict): Metadata payload.

        Returns:
            Number of points upserted.
        """
        if not chunks:
            return 0

        points = []
        for chunk in chunks:
            point_id = _deterministic_uuid(chunk["chunk_id"])
            points.append(
                PointStruct(
                    id=point_id,
                    vector=chunk["vector"],
                    payload=chunk["payload"],
                )
            )

        batch_size = 100
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            await self._client.upsert(
                collection_name=self._collection_name,
                points=batch,
            )

        log.info("qdrant_chunks_upserted", count=len(points))
        return len(points)

    async def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run vector similarity search with optional payload filters.

        Args:
            query_vector: Query embedding vector.
            top_k: Maximum results to return.
            filters: Optional dict with filter keys matching payload fields.

        Returns:
            List of dicts with 'chunk_id', 'score', and payload fields.
        """
        qdrant_filter = self._build_filter(filters) if filters else None

        results = await self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )

        hits: list[dict[str, Any]] = []
        for point in results.points:
            payload = point.payload or {}
            hits.append(
                {
                    "chunk_id": payload.get("chunk_id", str(point.id)),
                    "score": point.score,
                    **payload,
                }
            )

        return hits

    async def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all points belonging to a document."""
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )
        log.info("qdrant_document_deleted", doc_id=doc_id)

    async def get_stats(self) -> dict[str, Any]:
        """Return collection statistics."""
        try:
            info = await self._client.get_collection(self._collection_name)
            return {
                "total_points": info.points_count or 0,
                "total_documents": 0,  # Computed from unique doc_ids if needed
            }
        except Exception:
            return {"total_points": 0, "total_documents": 0}

    async def close(self) -> None:
        """Close the Qdrant client."""
        await self._client.close()

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_filter(filters: dict[str, Any]) -> Filter | None:
        """Build a Qdrant filter from a dict of field conditions."""
        conditions = []

        if filters.get("source_types"):
            conditions.append(
                FieldCondition(
                    key="source_type",
                    match=MatchAny(any=filters["source_types"]),
                )
            )

        if filters.get("companies"):
            conditions.append(
                FieldCondition(
                    key="company",
                    match=MatchAny(any=filters["companies"]),
                )
            )

        if filters.get("topics"):
            conditions.append(
                FieldCondition(
                    key="topics",
                    match=MatchAny(any=filters["topics"]),
                )
            )

        if not conditions:
            return None

        return Filter(must=conditions)
