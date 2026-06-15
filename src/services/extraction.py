import json
import re
from typing import Any

from core.logging import get_logger
from llm.base import LLMProvider

log = get_logger("extraction")


def _extract_json_payload(text: str) -> Any:
    """Extract a JSON object or array from LLM output that may contain markdown fences."""
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find the first JSON object or array
    for pattern in [r"\{[\s\S]*\}", r"\[[\s\S]*\]"]:
        match = re.search(pattern, cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue

    return {}


class ExtractionService:
    """LLM-powered entity and relationship extraction.

    Extraction prompts are dynamically built from the configured entity types
    and relationship types, making the service fully schema-configurable.
    """

    def __init__(
        self,
        llm: LLMProvider,
        entity_types: list[str],
        relationship_types: list[str],
    ) -> None:
        self._llm = llm
        self._entity_types = entity_types
        self._relationship_types = relationship_types

    async def extract_entities(self, text: str) -> list[dict[str, str]]:
        """Extract named entities from document text.

        Args:
            text: Full document text (may be truncated for large documents).

        Returns:
            List of dicts with 'name' and 'type' keys.
        """
        types_str = ", ".join(self._entity_types)
        system_prompt = (
            "You are an entity extraction system. Extract named entities from the text.\n\n"
            f"Allowed entity types: {types_str}\n\n"
            "Rules:\n"
            "- Only include clearly named entities explicitly present in the text.\n"
            "- Do not infer or invent entities that are not explicitly mentioned.\n"
            "- Choose the single most appropriate type for each entity.\n"
            "- Exclude generic phrases and vague concepts.\n"
            "- Deduplicate: if the same entity appears with different surface forms, "
            "use the most complete form.\n\n"
            'Return JSON: {"entities": [{"name": "...", "type": "..."}]}'
        )

        # Truncate very long texts to stay within context limits
        truncated = text[:12000] if len(text) > 12000 else text

        try:
            response = await self._llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": truncated},
                ],
                temperature=0.0,
            )
            payload = _extract_json_payload(response)
            entities = payload.get("entities", []) if isinstance(payload, dict) else []
            validated = self._validate_entities(entities)
            log.info("entities_extracted", count=len(validated))
            return validated
        except Exception as exc:
            log.warning("entity_extraction_failed", error=str(exc))
            return []

    async def extract_chunk_entities(self, chunk_text: str) -> list[dict[str, str]]:
        """Extract entities from a single chunk. Lighter prompt than document-level."""
        types_str = ", ".join(self._entity_types)
        system_prompt = (
            f"Extract named entities from this text chunk. Types: {types_str}. "
            "Only include clearly named entities. Return max 6 entities.\n"
            'Return JSON: {"entities": [{"name": "...", "type": "..."}]}'
        )

        try:
            response = await self._llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chunk_text},
                ],
                temperature=0.0,
            )
            payload = _extract_json_payload(response)
            entities = payload.get("entities", []) if isinstance(payload, dict) else []
            return self._validate_entities(entities)
        except Exception as exc:
            log.debug("chunk_entity_extraction_failed", error=str(exc))
            return []

    async def extract_relationships(
        self, text: str, entities: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Extract typed relationships between known entities.

        Args:
            text: Document text for context.
            entities: Previously extracted entities.

        Returns:
            List of dicts with 'subject', 'predicate', 'object',
            'subject_type', and 'object_type' keys.
        """
        if not entities:
            return []

        rel_types_str = ", ".join(self._relationship_types)
        entity_list = "\n".join(
            f"- {e['name']} ({e['type']})" for e in entities[:30]
        )

        system_prompt = (
            "You are a relationship extraction system. Given a list of known entities "
            "and a text, extract typed relationships between the entities.\n\n"
            f"Allowed relationship types: {rel_types_str}\n\n"
            "Rules:\n"
            "- Only extract relationships that are clearly stated or strongly implied.\n"
            "- Both subject and object must be from the provided entity list.\n"
            "- Use the exact entity names as provided.\n\n"
            "Return JSON: {\"relationships\": ["
            '{"subject": "...", "predicate": "...", "object": "...", '
            '"subject_type": "...", "object_type": "..."}'
            "]}"
        )

        user_prompt = f"Known entities:\n{entity_list}\n\nText:\n{text[:8000]}"

        try:
            response = await self._llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            payload = _extract_json_payload(response)
            relationships = (
                payload.get("relationships", []) if isinstance(payload, dict) else []
            )
            validated = self._validate_relationships(relationships)
            log.info("relationships_extracted", count=len(validated))
            return validated
        except Exception as exc:
            log.warning("relationship_extraction_failed", error=str(exc))
            return []

    # ── Validation ────────────────────────────────────────────────────────

    def _validate_entities(self, entities: Any) -> list[dict[str, str]]:
        """Filter and deduplicate extracted entities."""
        if not isinstance(entities, list):
            return []

        valid: list[dict[str, str]] = []
        seen: set[str] = set()
        allowed_types = set(self._entity_types)

        for entity in entities:
            if not isinstance(entity, dict):
                continue
            name = entity.get("name", "").strip()
            entity_type = entity.get("type", "")
            if not name or entity_type not in allowed_types:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            valid.append({"name": name, "type": entity_type})

        return valid

    def _validate_relationships(self, relationships: Any) -> list[dict[str, str]]:
        """Filter extracted relationships to allowed types."""
        if not isinstance(relationships, list):
            return []

        valid: list[dict[str, str]] = []
        allowed_types = set(self._relationship_types)

        for rel in relationships:
            if not isinstance(rel, dict):
                continue
            subject = rel.get("subject", "").strip()
            predicate = rel.get("predicate", "").strip()
            obj = rel.get("object", "").strip()
            if not subject or not predicate or not obj:
                continue
            if predicate not in allowed_types:
                continue
            valid.append(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "subject_type": rel.get("subject_type", ""),
                    "object_type": rel.get("object_type", ""),
                }
            )

        return valid
