from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

# ── Enums ─────────────────────────────────────────────────────────────────────


class DocumentStatus(StrEnum):
    pending = "pending"
    processing = "processing"
    processed = "processed"
    failed = "failed"


class GraphProcessStatus(StrEnum):
    pending = "pending"
    processing = "processing"
    processed = "processed"
    failed = "failed"
    skipped = "skipped"


class RetrievalProfile(StrEnum):
    rag = "rag"
    graph_rag = "graph_rag"
    hybrid = "hybrid"


# ── Health ────────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = Field(default="ok", description="Service health status.")
    neo4j: str = Field(default="unknown", description="Neo4j connectivity status.")
    qdrant: str = Field(default="unknown", description="Qdrant connectivity status.")


class Neo4jStats(BaseModel):
    total_nodes: int = Field(default=0, description="Total node count.")
    total_relationships: int = Field(default=0, description="Total relationship count.")
    nodes_by_label: dict[str, int] = Field(
        default_factory=dict, description="Node count grouped by label."
    )
    relationships_by_type: dict[str, int] = Field(
        default_factory=dict, description="Relationship count grouped by type."
    )


class QdrantStats(BaseModel):
    total_points: int = Field(default=0, description="Total indexed vector count.")
    total_documents: int = Field(default=0, description="Unique document count in vector store.")


class DetailedHealthResponse(BaseModel):
    status: str = Field(default="ok", description="Service health status.")
    neo4j: Neo4jStats = Field(default_factory=Neo4jStats)
    qdrant: QdrantStats = Field(default_factory=QdrantStats)
    storage_path: str = Field(default="", description="Document storage path.")


# ── Documents ─────────────────────────────────────────────────────────────────


class DocumentMeta(BaseModel):
    doc_id: str = Field(description="Stable document identifier.")
    file_name: str = Field(description="Original uploaded filename.")
    content_type: str = Field(description="MIME type of the uploaded file.")
    size_bytes: int = Field(description="Size of the uploaded raw file in bytes.")
    status: DocumentStatus = Field(description="Current processing state.")
    created_at: datetime = Field(description="Metadata creation timestamp.")
    updated_at: datetime = Field(description="Last metadata update timestamp.")
    title: str | None = Field(default=None, description="Human-readable document title.")
    source_url: str | None = Field(default=None, description="Original source URL.")
    source_type: str | None = Field(default=None, description="Source type label.")
    company: str | None = Field(default=None, description="Associated company.")
    speaker: str | None = Field(default=None, description="Primary speaker or author.")
    published_at: datetime | None = Field(default=None, description="Publication timestamp.")
    language: str | None = Field(default=None, description="Language code.")
    topics: list[str] = Field(default_factory=list, description="Topic labels.")
    entities: list[str] = Field(default_factory=list, description="Named entities.")
    keywords: list[str] = Field(default_factory=list, description="Keywords.")
    summary: str | None = Field(default=None, description="Document-level summary.")
    error: str | None = Field(default=None, description="Last processing error.")
    chunk_count: int | None = Field(default=None, description="Number of indexed chunks.")
    graph_status: GraphProcessStatus | None = Field(
        default=None, description="Graph extraction status."
    )
    graph_error: str | None = Field(default=None, description="Graph extraction error.")
    skipped: bool = Field(default=False, description="True when document already existed.")


class DocumentListResponse(BaseModel):
    items: list[DocumentMeta] = Field(description="Document metadata records.")
    total: int = Field(description="Total number of stored documents.")


class DocumentStatusResponse(BaseModel):
    doc_id: str = Field(description="Document identifier.")
    status: DocumentStatus = Field(description="Current processing state.")
    error: str | None = Field(default=None, description="Processing error.")
    chunk_count: int | None = Field(default=None, description="Chunk count after processing.")
    graph_status: GraphProcessStatus | None = Field(default=None, description="Graph status.")
    graph_error: str | None = Field(default=None, description="Graph processing error.")


class BatchFileResult(BaseModel):
    doc_id: str = Field(description="Document identifier.")
    file_name: str = Field(description="Original filename.")
    accepted: bool = Field(description="Whether the file was accepted for processing.")
    skipped: bool = Field(default=False, description="True if document already existed.")
    error: str | None = Field(default=None, description="Error message if not accepted.")


class BatchIngestResponse(BaseModel):
    total_files: int = Field(description="Total number of files received.")
    accepted: int = Field(description="Number of files accepted for processing.")
    skipped: int = Field(description="Number of files skipped (already exist).")
    failed: int = Field(description="Number of files that failed validation.")
    results: list[BatchFileResult] = Field(description="Per-file results.")


# ── Search ────────────────────────────────────────────────────────────────────


class SearchFilters(BaseModel):
    source_types: list[str] = Field(default_factory=list, description="Source type filter.")
    companies: list[str] = Field(default_factory=list, description="Company filter.")
    topics: list[str] = Field(default_factory=list, description="Topic filter (OR semantics).")
    date_from: datetime | None = Field(default=None, description="Lower bound for published_at.")
    date_to: datetime | None = Field(default=None, description="Upper bound for published_at.")


class GraphOptions(BaseModel):
    depth: int = Field(default=2, ge=1, le=4, description="Graph traversal depth.")
    max_matched_entities: int = Field(
        default=10, ge=1, le=100, description="Max query-matched entities."
    )
    max_graph_related: int = Field(
        default=100, ge=1, le=500, description="Max related entities from graph."
    )
    max_expanded_chunks: int = Field(
        default=6, ge=1, le=50, description="Max graph-expanded chunk candidates."
    )
    include_relationships: bool = Field(
        default=False, description="Include sample relationship paths."
    )


class SearchRequest(BaseModel):
    queries: list[str] = Field(
        default_factory=list, description="Natural-language search queries."
    )
    retrieval_profile: RetrievalProfile = Field(
        default=RetrievalProfile.rag, description="Retrieval strategy."
    )
    top_k: int = Field(default=10, ge=1, le=100, description="Max results per query.")
    filters: SearchFilters = Field(default_factory=SearchFilters, description="Optional filters.")
    graph_options: GraphOptions = Field(
        default_factory=GraphOptions, description="Graph traversal options."
    )
    debug: bool = Field(default=False, description="Include debug metadata in response.")

    @model_validator(mode="after")
    def validate_queries(self) -> "SearchRequest":
        queries = [q.strip() for q in self.queries if isinstance(q, str) and q.strip()]
        if not queries:
            raise ValueError("Provide at least one query in `queries`.")
        self.queries = queries
        return self


class SearchHit(BaseModel):
    chunk_id: str = Field(description="Indexed chunk identifier.")
    doc_id: str = Field(description="Source document identifier.")
    title: str | None = Field(default=None, description="Document title.")
    source_type: str | None = Field(default=None, description="Source type.")
    company: str | None = Field(default=None, description="Associated company.")
    speaker: str | None = Field(default=None, description="Speaker or author.")
    published_at: datetime | None = Field(default=None, description="Publication timestamp.")
    topics: list[str] = Field(default_factory=list, description="Associated topics.")
    summary: str | None = Field(default=None, description="Chunk or document summary.")
    content: str = Field(description="Full chunk text.")
    score: float | None = Field(default=None, description="Vector similarity score.")
    rerank_score: float | None = Field(default=None, description="Final reranked score.")
    retrieval_reasons: list[str] = Field(
        default_factory=list, description="Why this chunk was retrieved."
    )


class GraphResult(BaseModel):
    mode: RetrievalProfile = Field(description="Retrieval profile used.")
    matched_entities: list[str] = Field(
        default_factory=list, description="Query entities matched in the graph."
    )
    related_entities: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Related entities grouped by type (e.g. {'Organization': ['Acme']}).",
    )
    sample_relationships: list[str] = Field(
        default_factory=list, description="Sample relationship paths."
    )
    total_related: int = Field(default=0, description="Total related entities.")
    applied: bool = Field(default=False, description="Whether graph contributed to retrieval.")


class SearchDebug(BaseModel):
    original_query: str = Field(description="Original user query.")
    effective_search_text: str = Field(description="Final search text after graph expansion.")
    matched_entities: list[str] = Field(default_factory=list, description="Matched entity names.")
    expansion_terms: list[str] = Field(default_factory=list, description="Graph expansion terms.")


class QueryResult(BaseModel):
    query: str = Field(description="Original query.")
    items: list[SearchHit] = Field(description="Retrieved search hits.")
    total: int = Field(description="Number of returned hits.")
    graph: GraphResult | None = Field(default=None, description="Graph retrieval result.")
    debug: SearchDebug | None = Field(default=None, description="Debug metadata.")
    answer: str | None = Field(default=None, description="Natively synthesized LLM answer.")


class SearchResponse(BaseModel):
    results: list[QueryResult] = Field(description="One result per input query.")
    total_queries: int = Field(description="Number of executed queries.")


# ── Graph Inspection ──────────────────────────────────────────────────────────


class GraphEntitySummary(BaseModel):
    name: str = Field(description="Entity display name.")
    label: str = Field(description="Entity type label.")
    normalized_name: str = Field(description="Normalized name for matching.")


class GraphEntityListResponse(BaseModel):
    items: list[GraphEntitySummary] = Field(description="Entity nodes.")
    total: int = Field(description="Total entity count.")


class GraphDocumentSummary(BaseModel):
    doc_id: str = Field(description="Document node identifier.")
    entity_count: int = Field(default=0, description="Connected entity count.")
    chunk_count: int = Field(default=0, description="Connected chunk count.")


class GraphDocumentListResponse(BaseModel):
    items: list[GraphDocumentSummary] = Field(description="Document nodes.")
    total: int = Field(description="Total document count in graph.")


class GraphDocumentDetail(BaseModel):
    doc_id: str = Field(description="Document node identifier.")
    entities: list[GraphEntitySummary] = Field(
        default_factory=list, description="Connected entities."
    )
    chunk_count: int = Field(default=0, description="Connected chunk count.")
    relationships: list[str] = Field(
        default_factory=list, description="Sample relationships."
    )


class GraphStatsResponse(BaseModel):
    total_nodes: int = Field(default=0, description="Total node count.")
    total_relationships: int = Field(default=0, description="Total relationship count.")
    nodes_by_label: dict[str, int] = Field(
        default_factory=dict, description="Nodes by label."
    )
    relationships_by_type: dict[str, int] = Field(
        default_factory=dict, description="Relationships by type."
    )
