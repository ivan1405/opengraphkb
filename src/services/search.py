import difflib
import re
from typing import Any

from core.config import Settings
from core.logging import get_logger
from llm.base import LLMProvider
from models.schemas import (
    GraphOptions,
    GraphResult,
    QueryResult,
    RetrievalProfile,
    SearchDebug,
    SearchFilters,
    SearchHit,
)
from services.neo4j import Neo4jService, _normalize_name
from services.qdrant import QdrantService

log = get_logger("search")

_MIN_QUERY_ENTITY_TOKEN_LENGTH = 3
_MIN_FUZZY_RATIO = 0.88
_MAX_EXPANSION_TERMS = 8


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _fuzzy_ratio(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, left, right).ratio()


class SearchService:
    """Unified search service supporting RAG, GraphRAG, and Hybrid retrieval.

    Each retrieval profile controls how much graph context is used:
    - ``rag``: Pure vector similarity search through Qdrant.
    - ``graph_rag``: Vector search + graph entity matching and metadata.
    - ``hybrid``: Vector search + graph entity expansion + chunk expansion + reranking.
    """

    def __init__(
        self,
        llm: LLMProvider,
        neo4j: Neo4jService,
        qdrant: QdrantService,
        config: Settings,
    ) -> None:
        self._llm = llm
        self._neo4j = neo4j
        self._qdrant = qdrant
        self._config = config

    async def search(
        self,
        query: str,
        *,
        profile: RetrievalProfile = RetrievalProfile.rag,
        top_k: int = 10,
        filters: SearchFilters | None = None,
        graph_options: GraphOptions | None = None,
        debug: bool = False,
    ) -> QueryResult:
        """Execute a search query with the specified retrieval profile."""
        graph_opts = graph_options or GraphOptions()
        search_filters = filters or SearchFilters()

        # ── Step 1: Embed the query ───────────────────────────────────────
        query_vectors = await self._llm.embed([query])
        query_vector = query_vectors[0]

        matched_entities: list[str] = []
        related_entities: dict[str, list[str]] = {}
        sample_relationships: list[str] = []
        expansion_terms: list[str] = []
        effective_search = query

        # ── Step 2: Graph context (for graph_rag and hybrid) ──────────────
        graph_enabled = profile != RetrievalProfile.rag

        if graph_enabled:
            matched_entities = await self._match_query_entities(
                query, max_matches=graph_opts.max_matched_entities
            )

            if matched_entities:
                expansion_result = await self._neo4j.expand_entities(
                    matched_entities,
                    depth=graph_opts.depth,
                    max_related=graph_opts.max_graph_related,
                    target_labels=self._config.entity_type_list,
                    include_relationships=graph_opts.include_relationships,
                )
                related_entities = expansion_result.get("entities", {})
                sample_relationships = expansion_result.get("relationships", [])

                # Build expansion terms from related entity names
                source_names = {_normalize_name(n) for n in matched_entities}
                seen: set[str] = set()
                for names in related_entities.values():
                    for name in names:
                        normalized = _normalize_name(name)
                        if normalized not in source_names and normalized not in seen:
                            seen.add(normalized)
                            expansion_terms.append(name)

                expansion_terms = expansion_terms[:_MAX_EXPANSION_TERMS]
                if expansion_terms:
                    effective_search = query + " " + " ".join(expansion_terms)

        # ── Step 3: Vector search ─────────────────────────────────────────
        qdrant_filters: dict[str, Any] = {}
        if search_filters.source_types:
            qdrant_filters["source_types"] = search_filters.source_types
        if search_filters.companies:
            qdrant_filters["companies"] = search_filters.companies
        if search_filters.topics:
            qdrant_filters["topics"] = search_filters.topics

        raw_hits = await self._qdrant.search(
            query_vector=query_vector,
            top_k=top_k,
            filters=qdrant_filters if qdrant_filters else None,
        )

        # ── Step 4: Build search hits ─────────────────────────────────────
        items: list[SearchHit] = []
        for hit in raw_hits:
            items.append(
                SearchHit(
                    chunk_id=hit.get("chunk_id", ""),
                    doc_id=hit.get("doc_id", ""),
                    title=hit.get("title"),
                    source_type=hit.get("source_type"),
                    company=hit.get("company"),
                    speaker=hit.get("speaker"),
                    published_at=hit.get("published_at"),
                    topics=hit.get("topics", []),
                    summary=hit.get("summary"),
                    content=hit.get("content", ""),
                    score=hit.get("score"),
                    rerank_score=hit.get("score"),
                    retrieval_reasons=["base_retrieval"],
                )
            )

        # ── Step 5: Hybrid reranking (hybrid profile only) ────────────────
        if profile == RetrievalProfile.hybrid and matched_entities and items:
            seed_ids = [item.chunk_id for item in items[:3]]
            graph_signals = await self._neo4j.get_related_chunks(
                seed_ids, matched_entities,
                max_candidates=graph_opts.max_expanded_chunks,
            )

            if graph_signals:
                max_score = max((item.score or 0.0) for item in items) or 1.0
                for item in items:
                    base_normalized = (item.score or 0.0) / max_score
                    signal = graph_signals.get(item.chunk_id, {})
                    graph_bonus = (
                        signal.get("entity_overlap", 0.0) * 0.8
                        + signal.get("graph_proximity", 0.0) * 0.6
                        + signal.get("adjacency_bonus", 0.0) * 0.1
                    )
                    item.rerank_score = round(0.6 * base_normalized + graph_bonus, 4)
                    reasons = signal.get("reasons", [])
                    if reasons:
                        item.retrieval_reasons.extend(reasons)

                items.sort(
                    key=lambda x: (x.rerank_score or 0.0, x.score or 0.0),
                    reverse=True,
                )

        # ── Step 6: Build response ────────────────────────────────────────
        graph_result = None
        if graph_enabled:
            total_related = sum(len(v) for v in related_entities.values())
            graph_result = GraphResult(
                mode=profile,
                matched_entities=matched_entities,
                related_entities=related_entities,
                sample_relationships=sample_relationships,
                total_related=total_related,
                applied=bool(expansion_terms or related_entities),
            )

        debug_info = None
        if debug:
            debug_info = SearchDebug(
                original_query=query,
                effective_search_text=effective_search,
                matched_entities=matched_entities,
                expansion_terms=expansion_terms,
            )

        return QueryResult(
            query=query,
            items=items[:top_k],
            total=len(items),
            graph=graph_result,
            debug=debug_info,
        )

    # ── Query Entity Matching ─────────────────────────────────────────────

    async def _match_query_entities(
        self, query: str, *, max_matches: int = 10
    ) -> list[str]:
        """Match query text against known graph entity names."""
        try:
            known = await self._neo4j.get_entity_names()
            query_normalized = _normalize_name(query)
            query_tokens = set(_tokenize(query))

            candidates: list[tuple[float, str]] = []
            seen: set[str] = set()

            for entry in known:
                name = entry.get("name", "")
                normalized = entry.get("normalized_name", "")
                if not normalized or len(normalized) < _MIN_QUERY_ENTITY_TOKEN_LENGTH:
                    continue
                if normalized in seen:
                    continue

                score = self._score_entity_match(query_normalized, query_tokens, entry)
                if score > 0:
                    seen.add(normalized)
                    candidates.append((score, name))

            candidates.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
            matched = [name for _, name in candidates[:max_matches]]

            if matched:
                log.info(
                    "query_entities_matched",
                    query=query,
                    count=len(matched),
                    entities=matched[:5],
                )

            return matched
        except Exception as exc:
            log.warning("query_entity_match_failed", query=query, error=str(exc))
            return []

    @staticmethod
    def _score_entity_match(
        query_normalized: str,
        query_tokens: set[str],
        entry: dict[str, str],
    ) -> float:
        """Score how well an entity matches the query."""
        normalized_name = entry.get("normalized_name", "")
        name_tokens = set(_tokenize(entry.get("name", "")))

        # Exact substring match
        if normalized_name in query_normalized:
            return 1.0

        # Token overlap
        significant_tokens = {
            t for t in name_tokens if len(t) >= _MIN_QUERY_ENTITY_TOKEN_LENGTH
        }
        if significant_tokens and significant_tokens.issubset(query_tokens):
            return 0.95

        # Single token match for multi-token names
        overlap = significant_tokens & query_tokens
        if overlap and len(overlap) >= 1 and len(significant_tokens) <= 3:
            return 0.8

        # Fuzzy matching
        if len(normalized_name) >= _MIN_QUERY_ENTITY_TOKEN_LENGTH:
            for token in query_tokens:
                if len(token) < _MIN_QUERY_ENTITY_TOKEN_LENGTH:
                    continue
                if abs(len(token) - len(normalized_name)) > 2:
                    continue
                ratio = _fuzzy_ratio(token, normalized_name)
                if ratio >= _MIN_FUZZY_RATIO:
                    return ratio

        return 0.0
