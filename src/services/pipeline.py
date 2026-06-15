import asyncio
from datetime import UTC, datetime
from typing import Any

from chunking.base import ChunkingStrategy
from core.config import Settings
from core.logging import get_logger
from llm.base import LLMProvider
from models.schemas import DocumentStatus, GraphProcessStatus
from parsers.registry import ParserRegistry
from services.extraction import ExtractionService
from services.neo4j import Neo4jService, _normalize_name
from services.qdrant import QdrantService
from services.storage import StorageService

log = get_logger("pipeline")

_PIPELINE_SEMAPHORE = asyncio.Semaphore(2)


class IngestionPipeline:
    """Orchestrates document ingestion through parsing, chunking, embedding,
    vector upsert, entity extraction, and graph upsert.

    All dependencies are injected via constructor for testability and
    loose coupling.
    """

    def __init__(
        self,
        parser_registry: ParserRegistry,
        chunker: ChunkingStrategy,
        llm: LLMProvider,
        neo4j: Neo4jService,
        qdrant: QdrantService,
        storage: StorageService,
        extractor: ExtractionService,
        config: Settings,
    ) -> None:
        self._parsers = parser_registry
        self._chunker = chunker
        self._llm = llm
        self._neo4j = neo4j
        self._qdrant = qdrant
        self._storage = storage
        self._extractor = extractor
        self._config = config

    async def run(self, doc_id: str, file_name: str) -> None:
        """Execute the full ingestion pipeline for a document.

        This method is designed to be called from a BackgroundTask.
        """
        meta = await self._storage.get_metadata(doc_id)
        if meta is None:
            log.error("pipeline_metadata_missing", doc_id=doc_id)
            return

        log.info("pipeline_started", doc_id=doc_id)
        async with _PIPELINE_SEMAPHORE:
            await self._run(doc_id, file_name, meta)

    async def run_graph_only(self, doc_id: str) -> None:
        """Re-run only the graph extraction pipeline for a document."""
        meta = await self._storage.get_metadata(doc_id)
        if meta is None:
            log.error("pipeline_metadata_missing", doc_id=doc_id)
            return

        log.info("pipeline_graph_only_started", doc_id=doc_id)
        async with _PIPELINE_SEMAPHORE:
            await self._run_graph(doc_id, meta)

    async def _run(self, doc_id: str, file_name: str, meta: dict) -> None:
        try:
            # ── Stage 1: Update status ────────────────────────────────────
            meta["status"] = DocumentStatus.processing
            await self._storage.save_metadata(doc_id, meta)

            # ── Stage 2: Parse document ───────────────────────────────────
            file_bytes = await self._storage.download_document(doc_id, file_name)
            content_type = meta.get("content_type", "application/octet-stream")
            parser = self._parsers.get_parser(content_type, file_name)
            result = parser if isinstance(parser, str) else await parser.parse(file_bytes, file_name)

            text = result.text.strip()
            if not text:
                raise ValueError("No text could be extracted from the document")

            # ── Stage 3: Store normalized text ────────────────────────────
            await self._storage.upload_normalized(doc_id, text)
            log.info("document_parsed", doc_id=doc_id, chars=len(text))

            # ── Stage 4: Chunk ────────────────────────────────────────────
            chunks = self._chunker.chunk(text)
            if not chunks:
                raise ValueError("No chunks generated from the document")
            log.info("document_chunked", doc_id=doc_id, chunks=len(chunks))

            # ── Stage 5: Embed ────────────────────────────────────────────
            chunk_texts = [c.text for c in chunks]
            embeddings = await self._llm.embed(chunk_texts)
            log.info("document_embedded", doc_id=doc_id, vectors=len(embeddings))

            # ── Stage 6: Upsert to Qdrant ────────────────────────────────
            qdrant_chunks: list[dict[str, Any]] = []
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                chunk_id = f"{doc_id}_chunk_{chunk.index}"
                qdrant_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "vector": embedding,
                        "payload": {
                            "chunk_id": chunk_id,
                            "doc_id": doc_id,
                            "chunk_index": chunk.index,
                            "title": meta.get("title"),
                            "source_type": meta.get("source_type"),
                            "company": meta.get("company"),
                            "speaker": meta.get("speaker"),
                            "published_at": meta.get("published_at"),
                            "topics": meta.get("topics", []),
                            "entities": meta.get("entities", []),
                            "keywords": meta.get("keywords", []),
                            "summary": meta.get("summary"),
                            "content": chunk.text,
                        },
                    }
                )

            await self._qdrant.upsert_chunks(qdrant_chunks)

            # ── Stage 7: Update RAG status ────────────────────────────────
            meta["status"] = DocumentStatus.processed
            meta["chunk_count"] = len(chunks)
            meta["error"] = None
            meta["updated_at"] = datetime.now(UTC).isoformat()
            await self._storage.save_metadata(doc_id, meta)
            log.info("document_rag_processed", doc_id=doc_id, chunks=len(chunks))

            # ── Stage 8: Graph extraction ─────────────────────────────────
            if self._config.enable_graph_extraction:
                await self._run_graph(doc_id, meta, text=text, chunks=chunks)

        except Exception as exc:
            log.error("pipeline_failed", doc_id=doc_id, error=str(exc))
            meta["status"] = DocumentStatus.failed
            meta["error"] = str(exc)
            meta["updated_at"] = datetime.now(UTC).isoformat()
            await self._storage.save_metadata(doc_id, meta)

    async def _run_graph(
        self,
        doc_id: str,
        meta: dict,
        *,
        text: str | None = None,
        chunks: list | None = None,
    ) -> None:
        """Run graph extraction and upsert."""
        if not self._config.enable_graph_extraction:
            meta["graph_status"] = GraphProcessStatus.skipped
            await self._storage.save_metadata(doc_id, meta)
            return

        try:
            meta["graph_status"] = GraphProcessStatus.processing
            meta["graph_error"] = None
            await self._storage.save_metadata(doc_id, meta)

            # Load text if not provided
            if text is None:
                text = await self._storage.download_normalized(doc_id)
                if not text:
                    raise FileNotFoundError("Normalized text not found for graph extraction")

            # ── Create Document node ──────────────────────────────────────
            await self._neo4j.upsert_document(doc_id, {
                "title": meta.get("title", ""),
                "source_type": meta.get("source_type", ""),
                "company": meta.get("company", ""),
            })

            # ── Extract document-level entities ───────────────────────────
            entities = await self._extractor.extract_entities(text)
            log.info(
                "graph_entities_extracted",
                doc_id=doc_id,
                count=len(entities),
            )

            # ── Upsert entity nodes and MENTIONS edges ────────────────────
            for entity in entities:
                await self._neo4j.upsert_entity(entity["type"], entity["name"])
                await self._neo4j.create_relationship(
                    "Document", "doc_id", doc_id,
                    "MENTIONS",
                    "Entity", "normalized_name", _normalize_name(entity["name"]),
                )

            # ── Company BELONGS_TO edge ───────────────────────────────────
            company = meta.get("company")
            if company:
                await self._neo4j.upsert_entity("Organization", company)
                await self._neo4j.create_relationship(
                    "Document", "doc_id", doc_id,
                    "BELONGS_TO",
                    "Entity", "normalized_name", _normalize_name(company),
                )

            # ── Extract relationships ─────────────────────────────────────
            relationships = await self._extractor.extract_relationships(text, entities)
            for rel in relationships:
                subject_type = rel.get("subject_type", "Entity")
                object_type = rel.get("object_type", "Entity")
                # Ensure both entities exist
                await self._neo4j.upsert_entity(subject_type, rel["subject"])
                await self._neo4j.upsert_entity(object_type, rel["object"])
                await self._neo4j.create_relationship(
                    "Entity", "normalized_name", _normalize_name(rel["subject"]),
                    rel["predicate"],
                    "Entity", "normalized_name", _normalize_name(rel["object"]),
                )

            # ── Chunk-level extraction (optional) ─────────────────────────
            if self._config.enable_chunk_entity_extraction and chunks:
                for chunk in chunks:
                    chunk_id = f"{doc_id}_chunk_{chunk.index}"
                    chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)

                    await self._neo4j.upsert_chunk(chunk_id, doc_id, {
                        "chunk_index": chunk.index if hasattr(chunk, "index") else 0,
                    })

                    chunk_entities = await self._extractor.extract_chunk_entities(chunk_text)
                    for entity in chunk_entities:
                        await self._neo4j.upsert_entity(entity["type"], entity["name"])
                        await self._neo4j.create_relationship(
                            "Chunk", "chunk_id", chunk_id,
                            "MENTIONS",
                            "Entity", "normalized_name", _normalize_name(entity["name"]),
                        )

            # ── Update status ─────────────────────────────────────────────
            meta["graph_status"] = GraphProcessStatus.processed
            meta["graph_error"] = None
            meta["updated_at"] = datetime.now(UTC).isoformat()
            await self._storage.save_metadata(doc_id, meta)
            log.info(
                "graph_processed",
                doc_id=doc_id,
                entities=len(entities),
                relationships=len(relationships),
            )

        except Exception as exc:
            log.warning("graph_extraction_failed", doc_id=doc_id, error=str(exc))
            meta["graph_status"] = GraphProcessStatus.failed
            meta["graph_error"] = str(exc)
            meta["updated_at"] = datetime.now(UTC).isoformat()
            await self._storage.save_metadata(doc_id, meta)
