from fastapi import APIRouter, HTTPException

from core.logging import get_logger
from models.schemas import (
    GraphDocumentDetail,
    GraphDocumentListResponse,
    GraphDocumentSummary,
    GraphEntityListResponse,
    GraphEntitySummary,
    GraphStatsResponse,
)
from services.neo4j import Neo4jService

log = get_logger("graph_api")

router = APIRouter(prefix="/graph", tags=["graph"])

_neo4j: Neo4jService | None = None


def init_graph(neo4j: Neo4jService) -> None:
    """Wire up the Neo4j service. Called during app startup."""
    global _neo4j
    _neo4j = neo4j


def _get_neo4j() -> Neo4jService:
    if _neo4j is None:
        raise RuntimeError("Neo4jService not initialized")
    return _neo4j


@router.get(
    "/stats",
    response_model=GraphStatsResponse,
    summary="Graph Stats",
    description="Returns node and relationship counts grouped by label and type.",
)
async def graph_stats() -> GraphStatsResponse:
    neo4j = _get_neo4j()
    stats = await neo4j.get_graph_stats()
    return GraphStatsResponse(**stats)


@router.get(
    "/entities",
    response_model=GraphEntityListResponse,
    summary="List Entities",
    description="Returns all entity nodes in the graph.",
)
async def list_entities() -> GraphEntityListResponse:
    neo4j = _get_neo4j()
    entities = await neo4j.get_entity_list()
    items = [GraphEntitySummary(**e) for e in entities]
    return GraphEntityListResponse(items=items, total=len(items))


@router.get(
    "/documents",
    response_model=GraphDocumentListResponse,
    summary="List Graph Documents",
    description="Returns document nodes with entity and chunk counts.",
)
async def list_graph_documents() -> GraphDocumentListResponse:
    neo4j = _get_neo4j()
    docs = await neo4j.get_document_summaries()
    items = [GraphDocumentSummary(**d) for d in docs]
    return GraphDocumentListResponse(items=items, total=len(items))


@router.get(
    "/documents/{doc_id}",
    response_model=GraphDocumentDetail,
    summary="Graph Document Detail",
    description="Returns detailed graph information for a single document.",
    responses={404: {"description": "Document not found in graph."}},
)
async def get_graph_document(doc_id: str) -> GraphDocumentDetail:
    neo4j = _get_neo4j()
    detail = await neo4j.get_document_detail(doc_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found in graph")

    entities = [GraphEntitySummary(**e) for e in detail.get("entities", [])]
    return GraphDocumentDetail(
        doc_id=detail["doc_id"],
        entities=entities,
        chunk_count=detail.get("chunk_count", 0),
    )
