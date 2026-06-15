from fastapi import APIRouter

from core.logging import get_logger
from models.schemas import DetailedHealthResponse, HealthResponse, Neo4jStats, QdrantStats
from services.neo4j import Neo4jService
from services.qdrant import QdrantService

log = get_logger("health")

router = APIRouter(tags=["ops"])

# These will be set during app lifespan
_neo4j: Neo4jService | None = None
_qdrant: QdrantService | None = None
_storage_path: str = ""


def init_health(neo4j: Neo4jService, qdrant: QdrantService, storage_path: str) -> None:
    """Wire up service references for health checks. Called during app startup."""
    global _neo4j, _qdrant, _storage_path
    _neo4j = neo4j
    _qdrant = qdrant
    _storage_path = storage_path


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns service health including Neo4j and Qdrant connectivity.",
)
async def health() -> HealthResponse:
    neo4j_status = "unknown"
    qdrant_status = "unknown"

    if _neo4j:
        neo4j_status = "ok" if await _neo4j.health_check() else "error"
    if _qdrant:
        qdrant_status = "ok" if await _qdrant.health_check() else "error"

    overall = "ok" if neo4j_status == "ok" and qdrant_status == "ok" else "degraded"
    return HealthResponse(status=overall, neo4j=neo4j_status, qdrant=qdrant_status)


@router.get(
    "/health/detailed",
    response_model=DetailedHealthResponse,
    summary="Detailed Health",
    description="Returns detailed stats from Neo4j and Qdrant.",
)
async def health_detailed() -> DetailedHealthResponse:
    neo4j_stats = Neo4jStats()
    qdrant_stats = QdrantStats()

    if _neo4j:
        try:
            stats = await _neo4j.get_graph_stats()
            neo4j_stats = Neo4jStats(
                total_nodes=stats["total_nodes"],
                total_relationships=stats["total_relationships"],
                nodes_by_label=stats["nodes_by_label"],
                relationships_by_type=stats["relationships_by_type"],
            )
        except Exception as exc:
            log.warning("neo4j_stats_failed", error=str(exc))

    if _qdrant:
        try:
            stats = await _qdrant.get_stats()
            qdrant_stats = QdrantStats(
                total_points=stats["total_points"],
                total_documents=stats["total_documents"],
            )
        except Exception as exc:
            log.warning("qdrant_stats_failed", error=str(exc))

    return DetailedHealthResponse(
        neo4j=neo4j_stats,
        qdrant=qdrant_stats,
        storage_path=_storage_path,
    )
