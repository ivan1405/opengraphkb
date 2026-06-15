import re
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase

from core.logging import get_logger

log = get_logger("neo4j_service")


def _normalize_name(name: str) -> str:
    """Lowercase and collapse whitespace for deterministic matching."""
    return re.sub(r"\s+", " ", name.lower().strip())


class Neo4jService:
    """Graph service backed by Neo4j using native Cypher queries.

    Entities use dual labels: every entity node has both ``Entity`` (for
    uniform querying) and its specific type label (e.g. ``Person``).
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Create the async driver and verify connectivity."""
        self._driver = AsyncGraphDatabase.driver(
            self._uri, auth=(self._user, self._password)
        )
        await self._driver.verify_connectivity()
        await self._ensure_constraints()
        log.info("neo4j_connected", uri=self._uri)

    async def close(self) -> None:
        """Close the driver and release resources."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            log.info("neo4j_disconnected")

    async def health_check(self) -> bool:
        """Return True if Neo4j is reachable."""
        driver = self._get_driver()
        try:
            await driver.verify_connectivity()
            return True
        except Exception:
            return False

    # ── Schema ────────────────────────────────────────────────────────────

    async def _ensure_constraints(self) -> None:
        """Create uniqueness constraints if they don't exist."""
        driver = self._get_driver()
        constraints = [
            "CREATE CONSTRAINT entity_normalized_name IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.normalized_name IS UNIQUE",
            "CREATE CONSTRAINT document_doc_id IF NOT EXISTS "
            "FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_chunk_id IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
        ]
        async with driver.session() as session:
            for constraint in constraints:
                try:
                    await session.run(constraint)
                except Exception as exc:
                    # Some Neo4j editions don't support all constraint types
                    log.debug("constraint_skipped", constraint=constraint, error=str(exc))

    # ── Entity Operations ─────────────────────────────────────────────────

    async def upsert_entity(
        self, label: str, name: str, properties: dict[str, Any] | None = None
    ) -> tuple[str, str]:
        """Create or update an entity node with dual labels (Entity + type).

        Returns:
            Tuple of (normalized_name, action) where action is 'created' or 'reused'.
        """
        driver = self._get_driver()
        normalized = _normalize_name(name)
        props = properties or {}

        query = f"""
        MERGE (e:Entity {{normalized_name: $normalized}})
        ON CREATE SET e.name = $name, e.created_at = datetime(), e += $props
        ON MATCH SET e.name = $name, e.updated_at = datetime(), e += $props
        WITH e,
             CASE WHEN e.created_at = e.updated_at OR e.updated_at IS NULL
                  THEN 'created' ELSE 'reused' END AS action
        SET e:{label}
        RETURN action
        """

        async with driver.session() as session:
            result = await session.run(
                query,
                normalized=normalized,
                name=name,
                props={**props, "label": label},
            )
            record = await result.single()
            action = record["action"] if record else "reused"

        return normalized, action

    async def upsert_document(self, doc_id: str, meta: dict[str, Any]) -> str:
        """Create or update a Document node.

        Returns:
            'created' or 'reused'.
        """
        driver = self._get_driver()
        safe_meta = {k: v for k, v in meta.items() if v is not None and isinstance(v, (str, int, float, bool))}

        query = """
        MERGE (d:Document {doc_id: $doc_id})
        ON CREATE SET d += $props, d.created_at = datetime()
        ON MATCH SET d += $props, d.updated_at = datetime()
        WITH d,
             CASE WHEN d.updated_at IS NULL THEN 'created' ELSE 'reused' END AS action
        RETURN action
        """

        async with driver.session() as session:
            result = await session.run(query, doc_id=doc_id, props=safe_meta)
            record = await result.single()
            return record["action"] if record else "reused"

    async def upsert_chunk(
        self, chunk_id: str, doc_id: str, properties: dict[str, Any] | None = None
    ) -> str:
        """Create or update a Chunk node and link it to its Document.

        Returns:
            'created' or 'reused'.
        """
        driver = self._get_driver()
        props = properties or {}
        safe_props = {k: v for k, v in props.items() if v is not None and isinstance(v, (str, int, float, bool))}

        query = """
        MERGE (c:Chunk {chunk_id: $chunk_id})
        ON CREATE SET c += $props, c.created_at = datetime()
        ON MATCH SET c += $props, c.updated_at = datetime()
        WITH c
        MATCH (d:Document {doc_id: $doc_id})
        MERGE (c)-[:PART_OF]->(d)
        RETURN CASE WHEN c.updated_at IS NULL THEN 'created' ELSE 'reused' END AS action
        """

        async with driver.session() as session:
            result = await session.run(
                query, chunk_id=chunk_id, doc_id=doc_id, props=safe_props
            )
            record = await result.single()
            return record["action"] if record else "reused"

    async def create_relationship(
        self,
        src_label: str,
        src_key: str,
        src_value: str,
        rel_type: str,
        tgt_label: str,
        tgt_key: str,
        tgt_value: str,
    ) -> str:
        """Create a typed relationship between two nodes.

        Args:
            src_label: Source node label (e.g. 'Document', 'Entity', 'Chunk').
            src_key: Source node property name to match on.
            src_value: Source node property value.
            rel_type: Relationship type (e.g. 'MENTIONS', 'BELONGS_TO').
            tgt_label: Target node label.
            tgt_key: Target node property name to match on.
            tgt_value: Target node property value.

        Returns:
            'created' or 'reused'.
        """
        driver = self._get_driver()

        query = f"""
        MATCH (a:{src_label} {{{src_key}: $src_value}})
        MATCH (b:{tgt_label} {{{tgt_key}: $tgt_value}})
        MERGE (a)-[r:{rel_type}]->(b)
        RETURN type(r) AS rel_type
        """

        async with driver.session() as session:
            result = await session.run(
                query, src_value=src_value, tgt_value=tgt_value
            )
            record = await result.single()
            return "created" if record else "reused"

    # ── Deletion ──────────────────────────────────────────────────────────

    async def delete_document_graph(self, doc_id: str) -> int:
        """Delete a document node and all its connected chunks and edges.

        Returns:
            Number of nodes deleted.
        """
        driver = self._get_driver()

        # First delete connected chunks
        query_chunks = """
        MATCH (d:Document {doc_id: $doc_id})<-[:PART_OF]-(c:Chunk)
        DETACH DELETE c
        RETURN count(c) AS deleted_chunks
        """

        # Then delete the document itself and its direct edges
        query_doc = """
        MATCH (d:Document {doc_id: $doc_id})
        DETACH DELETE d
        RETURN count(d) AS deleted_docs
        """

        total_deleted = 0
        async with driver.session() as session:
            result = await session.run(query_chunks, doc_id=doc_id)
            record = await result.single()
            total_deleted += record["deleted_chunks"] if record else 0

            result = await session.run(query_doc, doc_id=doc_id)
            record = await result.single()
            total_deleted += record["deleted_docs"] if record else 0

        log.info("graph_document_deleted", doc_id=doc_id, nodes_deleted=total_deleted)
        return total_deleted

    # ── Query ─────────────────────────────────────────────────────────────

    async def get_entity_names(self) -> list[dict[str, str]]:
        """Return all entity names for query-time matching."""
        driver = self._get_driver()

        query = """
        MATCH (e:Entity)
        RETURN e.name AS name, e.label AS label, e.normalized_name AS normalized_name
        ORDER BY e.name
        LIMIT 5000
        """

        async with driver.session() as session:
            result = await session.run(query)
            records = [record.data() async for record in result]
            return records

    async def expand_entities(
        self,
        entity_names: list[str],
        *,
        depth: int = 2,
        max_related: int = 100,
        target_labels: list[str] | None = None,
        include_relationships: bool = False,
    ) -> dict[str, Any]:
        """Traverse the graph from matched entities and return related entities.

        Args:
            entity_names: Starting entity names.
            depth: Maximum traversal depth.
            max_related: Maximum number of related entities to return.
            target_labels: Only return entities with these labels.
            include_relationships: Include sample relationship paths.

        Returns:
            Dict with 'entities' (grouped by label) and optional 'relationships'.
        """
        driver = self._get_driver()
        normalized_names = [_normalize_name(name) for name in entity_names]

        if not normalized_names:
            return {"entities": {}, "relationships": [], "total": 0}

        # Variable-length path traversal (no APOC dependency)
        label_clause = ""
        if target_labels:
            label_clause = "AND (" + " OR ".join(f"end:{lbl}" for lbl in target_labels) + ")"

        query = f"""
        MATCH (start:Entity)
        WHERE start.normalized_name IN $names
        MATCH path = (start)-[*1..{depth}]-(end:Entity)
        WHERE end.normalized_name <> start.normalized_name
              AND NOT end.normalized_name IN $names
              {label_clause}
        RETURN DISTINCT end.name AS name, end.label AS label, end.normalized_name AS normalized_name
        LIMIT $max_related
        """

        entities_by_label: dict[str, list[str]] = {}
        async with driver.session() as session:
            result = await session.run(
                query, names=normalized_names, max_related=max_related
            )
            async for record in result:
                label = record["label"] or "Unknown"
                name = record["name"]
                entities_by_label.setdefault(label, []).append(name)

        relationships: list[str] = []
        if include_relationships and normalized_names:
            rel_query = """
            MATCH (start:Entity)-[r]-(end:Entity)
            WHERE start.normalized_name IN $names
            RETURN start.name + ' -[' + type(r) + ']-> ' + end.name AS path
            LIMIT 20
            """
            async with driver.session() as session:
                result = await session.run(rel_query, names=normalized_names)
                relationships = [record["path"] async for record in result]

        total = sum(len(names) for names in entities_by_label.values())
        return {"entities": entities_by_label, "relationships": relationships, "total": total}

    async def get_related_chunks(
        self,
        seed_chunk_ids: list[str],
        entity_names: list[str],
        *,
        max_candidates: int = 12,
    ) -> dict[str, dict[str, Any]]:
        """Find chunks related to seed chunks via shared entities or adjacency.

        Returns:
            Dict mapping chunk_id to signal dicts with graph_proximity,
            entity_overlap, adjacency_bonus, and reasons.
        """
        driver = self._get_driver()
        if not seed_chunk_ids:
            return {}

        # entity_names used for shared-entity lookup via the graph query

        # Find chunks connected via shared entities
        query = """
        MATCH (seed:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(candidate:Chunk)
        WHERE seed.chunk_id IN $seeds
              AND NOT candidate.chunk_id IN $seeds
        WITH candidate, collect(DISTINCT e.name) AS shared_entities
        RETURN candidate.chunk_id AS chunk_id,
               size(shared_entities) AS entity_overlap,
               shared_entities
        ORDER BY entity_overlap DESC
        LIMIT $max_candidates
        """

        signals: dict[str, dict[str, Any]] = {}
        async with driver.session() as session:
            result = await session.run(
                query, seeds=seed_chunk_ids, max_candidates=max_candidates
            )
            async for record in result:
                chunk_id = record["chunk_id"]
                signals[chunk_id] = {
                    "graph_proximity": min(record["entity_overlap"] / 3.0, 1.0),
                    "entity_overlap": min(record["entity_overlap"] / 3.0, 1.0),
                    "adjacency_bonus": 0.0,
                    "reasons": ["shared_entity_chunk"],
                }

        return signals

    # ── Stats ─────────────────────────────────────────────────────────────

    async def get_graph_stats(self) -> dict[str, Any]:
        """Return node and relationship counts grouped by label/type."""
        driver = self._get_driver()

        async with driver.session() as session:
            # Node counts by label
            result = await session.run(
                "MATCH (n) RETURN labels(n) AS labels, count(n) AS cnt"
            )
            nodes_by_label: dict[str, int] = {}
            total_nodes = 0
            async for record in result:
                labels = record["labels"]
                count = record["cnt"]
                total_nodes += count
                for label in labels:
                    nodes_by_label[label] = nodes_by_label.get(label, 0) + count

            # Relationship counts by type
            result = await session.run(
                "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS cnt"
            )
            rels_by_type: dict[str, int] = {}
            total_rels = 0
            async for record in result:
                rel_type = record["rel_type"]
                count = record["cnt"]
                total_rels += count
                rels_by_type[rel_type] = count

        return {
            "total_nodes": total_nodes,
            "total_relationships": total_rels,
            "nodes_by_label": nodes_by_label,
            "relationships_by_type": rels_by_type,
        }

    async def get_document_summaries(self) -> list[dict[str, Any]]:
        """List all Document nodes with entity and chunk counts."""
        driver = self._get_driver()

        query = """
        MATCH (d:Document)
        OPTIONAL MATCH (d)<-[:PART_OF]-(c:Chunk)
        OPTIONAL MATCH (d)-[:MENTIONS]->(e:Entity)
        RETURN d.doc_id AS doc_id,
               count(DISTINCT c) AS chunk_count,
               count(DISTINCT e) AS entity_count
        ORDER BY d.doc_id
        """

        async with driver.session() as session:
            result = await session.run(query)
            return [record.data() async for record in result]

    async def get_document_detail(self, doc_id: str) -> dict[str, Any] | None:
        """Get detailed graph information for a single document."""
        driver = self._get_driver()

        query = """
        MATCH (d:Document {doc_id: $doc_id})
        OPTIONAL MATCH (d)-[:MENTIONS]->(e:Entity)
        OPTIONAL MATCH (d)<-[:PART_OF]-(c:Chunk)
        RETURN d.doc_id AS doc_id,
               collect(DISTINCT {name: e.name, label: e.label, normalized_name: e.normalized_name}) AS entities,
               count(DISTINCT c) AS chunk_count
        """

        async with driver.session() as session:
            result = await session.run(query, doc_id=doc_id)
            record = await result.single()
            if not record:
                return None

            entities = [
                e for e in record["entities"]
                if e.get("name") is not None
            ]

            return {
                "doc_id": record["doc_id"],
                "entities": entities,
                "chunk_count": record["chunk_count"],
            }

    async def get_entity_list(self, limit: int = 500) -> list[dict[str, str]]:
        """List entity nodes with pagination."""
        driver = self._get_driver()

        query = """
        MATCH (e:Entity)
        RETURN e.name AS name, e.label AS label, e.normalized_name AS normalized_name
        ORDER BY e.name
        LIMIT $limit
        """

        async with driver.session() as session:
            result = await session.run(query, limit=limit)
            return [record.data() async for record in result]

    # ── Internal ──────────────────────────────────────────────────────────

    def _get_driver(self) -> AsyncDriver:
        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialized. Call connect() first.")
        return self._driver
