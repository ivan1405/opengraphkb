from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.documents import init_documents
from api.documents import router as documents_router
from api.graph import init_graph
from api.graph import router as graph_router
from api.health import init_health
from api.health import router as health_router
from api.search import init_search
from api.search import router as search_router
from chunking.factory import create_chunking_strategy
from core.config import get_settings
from core.logging import get_logger, setup_logging
from llm.factory import create_llm_provider
from parsers.registry import ParserRegistry
from services.extraction import ExtractionService
from services.neo4j import Neo4jService
from services.pipeline import IngestionPipeline
from services.qdrant import QdrantService
from services.search import SearchService
from services.storage import StorageService

log = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Manage application lifecycle — connect to databases on startup,
    close connections on shutdown."""
    config = get_settings()
    setup_logging(config.log_level)

    log.info("starting", app_name=config.app_name, debug=config.debug)

    # ── Initialize services ───────────────────────────────────────────────
    storage = StorageService(config.storage_base_path)
    llm = create_llm_provider(config)
    chunker = create_chunking_strategy(config)
    parser_registry = ParserRegistry.default()

    neo4j = Neo4jService(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
    await neo4j.connect()

    qdrant = QdrantService(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key or None,
        collection_name=config.qdrant_collection,
        embedding_dimensions=config.llm_embedding_dimensions,
    )
    await qdrant.ensure_collection()

    extractor = ExtractionService(
        llm=llm,
        entity_types=config.entity_type_list,
        relationship_types=config.relationship_type_list,
    )

    pipeline = IngestionPipeline(
        parser_registry=parser_registry,
        chunker=chunker,
        llm=llm,
        neo4j=neo4j,
        qdrant=qdrant,
        storage=storage,
        extractor=extractor,
        config=config,
    )

    search = SearchService(llm=llm, neo4j=neo4j, qdrant=qdrant, config=config)

    # ── Wire up API routes ────────────────────────────────────────────────
    init_health(neo4j, qdrant, config.storage_base_path)
    init_documents(storage, pipeline, neo4j, qdrant, config.enable_graph_extraction)
    init_search(search)
    init_graph(neo4j)

    log.info("ready", app_name=config.app_name)
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    log.info("shutting_down")
    await neo4j.close()
    await qdrant.close()
    await llm.close()
    log.info("shutdown_complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    config = get_settings()

    app = FastAPI(
        title=config.app_name,
        description=(
            "OpenGraphKB — A vendor-agnostic knowledge base combining "
            "Neo4j (graph) and Qdrant (vector) for RAG, GraphRAG, and Hybrid retrieval."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routers
    app.include_router(health_router, prefix="/api")
    app.include_router(documents_router, prefix="/api")
    app.include_router(search_router, prefix="/api")
    app.include_router(graph_router, prefix="/api")

    return app


app = create_app()


def run() -> None:
    """Entry point for ``python -m main``."""
    config = get_settings()
    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level=config.log_level.lower(),
    )


if __name__ == "__main__":
    run()
