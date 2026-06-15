import re
from datetime import UTC, datetime
from typing import Annotated
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
    status,
)

from core.logging import get_logger
from models.schemas import (
    BatchFileResult,
    BatchIngestResponse,
    DocumentListResponse,
    DocumentMeta,
    DocumentStatus,
    DocumentStatusResponse,
    GraphProcessStatus,
)
from services.neo4j import Neo4jService
from services.pipeline import IngestionPipeline
from services.qdrant import QdrantService
from services.storage import StorageService

log = get_logger("documents_api")

router = APIRouter(prefix="/documents", tags=["documents"])

# These will be set during app lifespan
_storage: StorageService | None = None
_pipeline: IngestionPipeline | None = None
_neo4j: Neo4jService | None = None
_qdrant: QdrantService | None = None
_enable_graph: bool = True


def init_documents(
    storage: StorageService,
    pipeline: IngestionPipeline,
    neo4j: Neo4jService,
    qdrant: QdrantService,
    enable_graph: bool,
) -> None:
    """Wire up service references. Called during app startup."""
    global _storage, _pipeline, _neo4j, _qdrant, _enable_graph
    _storage = storage
    _pipeline = pipeline
    _neo4j = neo4j
    _qdrant = qdrant
    _enable_graph = enable_graph


def _get_storage() -> StorageService:
    if _storage is None:
        raise RuntimeError("StorageService not initialized")
    return _storage


def _get_pipeline() -> IngestionPipeline:
    if _pipeline is None:
        raise RuntimeError("IngestionPipeline not initialized")
    return _pipeline


def _document_meta_response(meta: dict) -> DocumentMeta:
    """Build a DocumentMeta from raw metadata dict."""
    return DocumentMeta(**meta)


def _document_status_response(doc_id: str, meta: dict) -> DocumentStatusResponse:
    return DocumentStatusResponse(
        doc_id=doc_id,
        status=meta["status"],
        error=meta.get("error"),
        chunk_count=meta.get("chunk_count"),
        graph_status=meta.get("graph_status"),
        graph_error=meta.get("graph_error"),
    )


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentMeta,
    summary="Ingest Document",
    description=(
        "Uploads a document and triggers background ingestion. Supports PDF, DOCX, "
        "Markdown, and plain text. The pipeline parses the file, chunks the text, "
        "generates embeddings, stores vectors in Qdrant, and extracts entities into Neo4j."
    ),
    responses={
        200: {"description": "Document already exists. Existing metadata returned with skipped=true."},
        202: {"description": "Document accepted and queued for background processing."},
    },
)
async def ingest_document(
    background_tasks: BackgroundTasks,
    response: Response,
    file: Annotated[UploadFile, File(description="The file to ingest.")],
    doc_id: Annotated[str | None, Form(description="Optional document ID. UUID generated if omitted.")] = None,
    title: Annotated[str | None, Form(description="Document title. Defaults to filename.")] = None,
    source_url: Annotated[str | None, Form(description="Original source URL.")] = None,
    source_type: Annotated[str | None, Form(description="Source type label.")] = None,
    company: Annotated[str | None, Form(description="Associated company.")] = None,
    speaker: Annotated[str | None, Form(description="Primary speaker or author.")] = None,
    published_at: Annotated[datetime | None, Form(description="Publication timestamp.")] = None,
    language: Annotated[str | None, Form(description="Language code.")] = None,
    topics: Annotated[list[str] | None, Form(description="Topic labels (repeated form field).")] = None,
    entities: Annotated[list[str] | None, Form(description="Named entities.")] = None,
    keywords: Annotated[list[str] | None, Form(description="Keywords.")] = None,
    summary: Annotated[str | None, Form(description="Optional precomputed summary.")] = None,
) -> DocumentMeta:
    storage = _get_storage()
    pipeline = _get_pipeline()

    if not file.filename:
        raise HTTPException(status_code=422, detail="Uploaded file must have a filename")

    doc_id = doc_id or str(uuid4())

    # Check for existing document
    existing = await storage.get_metadata(doc_id)
    if existing is not None:
        response.status_code = status.HTTP_200_OK
        existing["skipped"] = True
        return _document_meta_response(existing)

    file_bytes = await file.read()
    now = datetime.now(UTC).isoformat()

    meta: dict = {
        "doc_id": doc_id,
        "file_name": file.filename,
        "content_type": file.content_type or "application/octet-stream",
        "size_bytes": len(file_bytes),
        "status": DocumentStatus.pending,
        "created_at": now,
        "updated_at": now,
        "title": title or file.filename,
        "source_url": source_url,
        "source_type": source_type,
        "company": company,
        "speaker": speaker,
        "published_at": published_at.isoformat() if published_at else None,
        "language": language,
        "topics": topics or [],
        "entities": entities or [],
        "keywords": keywords or [],
        "summary": summary,
        "error": None,
        "chunk_count": None,
        "graph_status": GraphProcessStatus.pending if _enable_graph else GraphProcessStatus.skipped,
        "graph_error": None,
        "skipped": False,
    }

    await storage.upload_document(doc_id, file.filename, file_bytes, meta["content_type"])
    await storage.save_metadata(doc_id, meta)

    background_tasks.add_task(pipeline.run, doc_id, file.filename)
    log.info("document_ingested", doc_id=doc_id, file=file.filename)

    return _document_meta_response(meta)


def _sanitize_doc_id(filename: str) -> str:
    """Derive a stable doc_id from a filename."""
    name = filename.rsplit(".", 1)[0] if "." in filename else filename
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name).lower().strip("_")[:120]


@router.post(
    "/batch",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=BatchIngestResponse,
    summary="Batch Ingest Documents",
    description=(
        "Upload multiple files for ingestion in a single request. Each file is "
        "queued for background processing. Returns per-file status with doc_ids."
    ),
)
async def batch_ingest(
    background_tasks: BackgroundTasks,
    files: Annotated[
        list[UploadFile], File(description="Files to ingest (multipart upload).")
    ],
    source_type: Annotated[
        str | None, Form(description="Source type applied to all files.")
    ] = None,
    company: Annotated[
        str | None, Form(description="Company applied to all files.")
    ] = None,
) -> BatchIngestResponse:
    storage = _get_storage()
    pipeline = _get_pipeline()

    results: list[BatchFileResult] = []
    accepted_count = 0
    skipped_count = 0
    failed_count = 0

    log.info(
        "batch_ingest_started",
        total_files=len(files),
        source_type=source_type,
        company=company,
    )

    for idx, file in enumerate(files, start=1):
        filename = file.filename or f"unnamed_{idx}"
        doc_id = _sanitize_doc_id(filename)

        log.info(
            "batch_file_processing",
            file_number=idx,
            total=len(files),
            filename=filename,
            doc_id=doc_id,
        )

        # ── Check for existing document ───────────────────────────────
        existing = await storage.get_metadata(doc_id)
        if existing is not None:
            log.info(
                "batch_file_skipped",
                file_number=idx,
                filename=filename,
                doc_id=doc_id,
                reason="already_exists",
            )
            results.append(
                BatchFileResult(
                    doc_id=doc_id,
                    file_name=filename,
                    accepted=False,
                    skipped=True,
                )
            )
            skipped_count += 1
            continue

        # ── Read and validate ─────────────────────────────────────────
        try:
            file_bytes = await file.read()
            if not file_bytes:
                raise ValueError("Empty file")
        except Exception as exc:
            log.warning(
                "batch_file_read_failed",
                file_number=idx,
                filename=filename,
                error=str(exc),
            )
            results.append(
                BatchFileResult(
                    doc_id=doc_id,
                    file_name=filename,
                    accepted=False,
                    error=f"Read error: {exc}",
                )
            )
            failed_count += 1
            continue

        # ── Build metadata and store ──────────────────────────────────
        now = datetime.now(UTC).isoformat()
        meta: dict = {
            "doc_id": doc_id,
            "file_name": filename,
            "content_type": file.content_type or "application/octet-stream",
            "size_bytes": len(file_bytes),
            "status": DocumentStatus.pending,
            "created_at": now,
            "updated_at": now,
            "title": filename,
            "source_url": None,
            "source_type": source_type,
            "company": company,
            "speaker": None,
            "published_at": None,
            "language": None,
            "topics": [],
            "entities": [],
            "keywords": [],
            "summary": None,
            "error": None,
            "chunk_count": None,
            "graph_status": (
                GraphProcessStatus.pending
                if _enable_graph
                else GraphProcessStatus.skipped
            ),
            "graph_error": None,
            "skipped": False,
        }

        await storage.upload_document(doc_id, filename, file_bytes, meta["content_type"])
        await storage.save_metadata(doc_id, meta)
        background_tasks.add_task(pipeline.run, doc_id, filename)

        log.info(
            "batch_file_accepted",
            file_number=idx,
            filename=filename,
            doc_id=doc_id,
            size_bytes=len(file_bytes),
            content_type=meta["content_type"],
        )

        results.append(
            BatchFileResult(
                doc_id=doc_id,
                file_name=filename,
                accepted=True,
            )
        )
        accepted_count += 1

    log.info(
        "batch_ingest_completed",
        total_files=len(files),
        accepted=accepted_count,
        skipped=skipped_count,
        failed=failed_count,
    )

    return BatchIngestResponse(
        total_files=len(files),
        accepted=accepted_count,
        skipped=skipped_count,
        failed=failed_count,
        results=results,
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List Documents",
    description="Returns all stored document metadata records.",
)
async def list_documents() -> DocumentListResponse:
    storage = _get_storage()
    items_raw = await storage.list_metadata()
    items = [_document_meta_response(m) for m in items_raw]
    return DocumentListResponse(items=items, total=len(items))


@router.get(
    "/{doc_id}",
    response_model=DocumentMeta,
    summary="Get Document",
    description="Returns the full metadata record for a single document.",
    responses={404: {"description": "Document not found."}},
)
async def get_document(doc_id: str) -> DocumentMeta:
    storage = _get_storage()
    meta = await storage.get_metadata(doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    return _document_meta_response(meta)


@router.get(
    "/{doc_id}/status",
    response_model=DocumentStatusResponse,
    summary="Get Document Status",
    description="Returns the current processing status for a document.",
    responses={404: {"description": "Document not found."}},
)
async def get_document_status(doc_id: str) -> DocumentStatusResponse:
    storage = _get_storage()
    meta = await storage.get_metadata(doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    return _document_status_response(doc_id, meta)


@router.delete(
    "/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Document",
    description="Deletes a document from all stores (filesystem, Qdrant, Neo4j).",
    responses={404: {"description": "Document not found."}},
)
async def delete_document(doc_id: str) -> None:
    storage = _get_storage()
    meta = await storage.get_metadata(doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

    await storage.delete_document(doc_id)
    await storage.delete_normalized(doc_id)
    await storage.delete_metadata(doc_id)

    if _qdrant:
        try:
            await _qdrant.delete_by_doc_id(doc_id)
        except Exception as exc:
            log.warning("qdrant_delete_failed", doc_id=doc_id, error=str(exc))

    if _neo4j and _enable_graph:
        try:
            await _neo4j.delete_document_graph(doc_id)
        except Exception as exc:
            log.warning("graph_delete_failed", doc_id=doc_id, error=str(exc))

    log.info("document_deleted", doc_id=doc_id)


@router.post(
    "/{doc_id}/reprocess",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentStatusResponse,
    summary="Reprocess Document",
    description="Deletes existing indexed data and re-runs the full pipeline.",
    responses={
        202: {"description": "Reprocessing queued."},
        404: {"description": "Document not found."},
    },
)
async def reprocess_document(doc_id: str, background_tasks: BackgroundTasks) -> DocumentStatusResponse:
    storage = _get_storage()
    pipeline = _get_pipeline()
    meta = await storage.get_metadata(doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

    # Reset status
    meta["status"] = DocumentStatus.pending
    meta["error"] = None
    meta["chunk_count"] = None
    meta["graph_status"] = GraphProcessStatus.pending if _enable_graph else GraphProcessStatus.skipped
    meta["graph_error"] = None
    meta["updated_at"] = datetime.now(UTC).isoformat()
    await storage.save_metadata(doc_id, meta)

    # Clean existing data
    if _qdrant:
        try:
            await _qdrant.delete_by_doc_id(doc_id)
        except Exception as exc:
            log.warning("qdrant_delete_on_reprocess_failed", doc_id=doc_id, error=str(exc))

    if _neo4j and _enable_graph:
        try:
            await _neo4j.delete_document_graph(doc_id)
        except Exception as exc:
            log.warning("graph_delete_on_reprocess_failed", doc_id=doc_id, error=str(exc))

    background_tasks.add_task(pipeline.run, doc_id, meta["file_name"])
    log.info("document_reprocess_queued", doc_id=doc_id)

    return _document_status_response(doc_id, meta)


@router.post(
    "/{doc_id}/reprocess/graph",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentStatusResponse,
    summary="Reprocess Document Graph",
    description="Re-runs only the graph extraction pipeline using stored normalized text.",
    responses={
        202: {"description": "Graph reprocessing queued."},
        404: {"description": "Document not found."},
        409: {"description": "Document not yet processed or graph extraction disabled."},
    },
)
async def reprocess_document_graph(doc_id: str, background_tasks: BackgroundTasks) -> DocumentStatusResponse:
    storage = _get_storage()
    pipeline = _get_pipeline()
    meta = await storage.get_metadata(doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

    if not _enable_graph:
        raise HTTPException(status_code=409, detail="Graph extraction is disabled")

    if meta.get("status") != DocumentStatus.processed:
        raise HTTPException(
            status_code=409,
            detail="Graph-only reprocess requires document to be fully processed first",
        )

    meta["graph_status"] = GraphProcessStatus.pending
    meta["graph_error"] = None
    meta["updated_at"] = datetime.now(UTC).isoformat()
    await storage.save_metadata(doc_id, meta)

    if _neo4j:
        try:
            await _neo4j.delete_document_graph(doc_id)
        except Exception as exc:
            log.warning("graph_delete_on_reprocess_failed", doc_id=doc_id, error=str(exc))

    background_tasks.add_task(pipeline.run_graph_only, doc_id)
    log.info("document_graph_reprocess_queued", doc_id=doc_id)

    return _document_status_response(doc_id, meta)
