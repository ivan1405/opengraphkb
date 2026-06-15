import asyncio
from typing import Annotated

from fastapi import APIRouter, Body

from core.logging import get_logger
from models.schemas import (
    RetrievalProfile,
    SearchRequest,
    SearchResponse,
)
from services.search import SearchService

log = get_logger("search_api")

router = APIRouter(prefix="/search", tags=["search"])

_search: SearchService | None = None


def init_search(search: SearchService) -> None:
    """Wire up the search service. Called during app startup."""
    global _search
    _search = search


def _get_search() -> SearchService:
    if _search is None:
        raise RuntimeError("SearchService not initialized")
    return _search


@router.post(
    "",
    response_model=SearchResponse,
    summary="Search Knowledge Base",
    description=(
        "Unified search endpoint supporting three retrieval profiles:\n\n"
        "- **rag**: Pure vector similarity search through Qdrant.\n"
        "- **graph_rag**: Vector search + graph entity matching and metadata.\n"
        "- **hybrid**: Vector + graph expansion + chunk expansion + reranking.\n\n"
        "Accepts one or more queries. Returns one result per query."
    ),
)
async def search(
    request: Annotated[
        SearchRequest,
        Body(
            openapi_examples={
                "rag_search": {
                    "summary": "RAG (vector only)",
                    "value": {
                        "queries": ["What are the latest AI developments?"],
                        "retrieval_profile": "rag",
                        "top_k": 5,
                    },
                },
                "graph_rag_search": {
                    "summary": "GraphRAG (vector + graph metadata)",
                    "value": {
                        "queries": ["How is OpenAI connected to AI safety?"],
                        "retrieval_profile": "graph_rag",
                        "top_k": 5,
                        "graph_options": {"depth": 2, "include_relationships": True},
                        "debug": True,
                    },
                },
                "hybrid_search": {
                    "summary": "Hybrid (full graph-aware retrieval)",
                    "value": {
                        "queries": ["What are the connections between AI companies?"],
                        "retrieval_profile": "hybrid",
                        "top_k": 10,
                        "graph_options": {
                            "depth": 2,
                            "max_expanded_chunks": 6,
                            "include_relationships": True,
                        },
                        "debug": True,
                    },
                },
            }
        ),
    ],
) -> SearchResponse:
    search_svc = _get_search()
    results = await asyncio.gather(
        *(
            search_svc.search(
                query,
                profile=request.retrieval_profile,
                top_k=request.top_k,
                filters=request.filters,
                graph_options=request.graph_options,
                debug=request.debug,
            )
            for query in request.queries
        )
    )
    return SearchResponse(results=list(results), total_queries=len(results))


@router.post(
    "/rag",
    response_model=SearchResponse,
    summary="RAG Search",
    description="Force pure vector-only search, ignoring graph context.",
)
async def search_rag(request: Annotated[SearchRequest, Body()]) -> SearchResponse:
    search_svc = _get_search()
    results = await asyncio.gather(
        *(
            search_svc.search(
                query,
                profile=RetrievalProfile.rag,
                top_k=request.top_k,
                filters=request.filters,
                debug=request.debug,
            )
            for query in request.queries
        )
    )
    return SearchResponse(results=list(results), total_queries=len(results))


@router.post(
    "/graph-rag",
    response_model=SearchResponse,
    summary="GraphRAG Search",
    description="Vector search enriched with graph entity matching and metadata.",
)
async def search_graph_rag(request: Annotated[SearchRequest, Body()]) -> SearchResponse:
    search_svc = _get_search()
    results = await asyncio.gather(
        *(
            search_svc.search(
                query,
                profile=RetrievalProfile.graph_rag,
                top_k=request.top_k,
                filters=request.filters,
                graph_options=request.graph_options,
                debug=request.debug,
            )
            for query in request.queries
        )
    )
    return SearchResponse(results=list(results), total_queries=len(results))


@router.post(
    "/hybrid",
    response_model=SearchResponse,
    summary="Hybrid Search",
    description="Full graph-aware retrieval with vector search + graph expansion + reranking.",
)
async def search_hybrid(request: Annotated[SearchRequest, Body()]) -> SearchResponse:
    search_svc = _get_search()
    results = await asyncio.gather(
        *(
            search_svc.search(
                query,
                profile=RetrievalProfile.hybrid,
                top_k=request.top_k,
                filters=request.filters,
                graph_options=request.graph_options,
                debug=request.debug,
            )
            for query in request.queries
        )
    )
    return SearchResponse(results=list(results), total_queries=len(results))
