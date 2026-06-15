from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    app_name: str = Field(default="OpenGraphKB")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="changeme_neo4j")

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str = Field(default="")
    qdrant_collection: str = Field(default="opengraphkb")

    # ── LLM Provider ──────────────────────────────────────────────────────────
    llm_provider: str = Field(default="openai")
    llm_api_key: str = Field(default="")
    llm_base_url: str = Field(default="https://api.openai.com/v1")
    llm_chat_model: str = Field(default="gpt-4o")
    llm_embedding_model: str = Field(default="text-embedding-3-small")
    llm_embedding_dimensions: int = Field(default=1536)

    # ── Storage ───────────────────────────────────────────────────────────────
    storage_base_path: str = Field(default="./data")

    # ── Ingestion ─────────────────────────────────────────────────────────────
    enable_graph_extraction: bool = Field(default=True)
    enable_chunk_entity_extraction: bool = Field(default=True)
    chunking_strategy: str = Field(default="fixed_size")
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=100)

    # ── Entity Schema ─────────────────────────────────────────────────────────
    entity_types: str = Field(
        default="Person,Organization,Project,Topic,Event,Technology,Location",
        description="Comma-separated list of allowed entity type labels for graph extraction.",
    )
    relationship_types: str = Field(
        default="MENTIONS,BELONGS_TO,RELATES_TO,LEADS,PRODUCES,LOCATED_IN,PART_OF",
        description="Comma-separated list of allowed relationship types for graph extraction.",
    )

    @property
    def entity_type_list(self) -> list[str]:
        """Return entity types as a list."""
        return [t.strip() for t in self.entity_types.split(",") if t.strip()]

    @property
    def relationship_type_list(self) -> list[str]:
        """Return relationship types as a list."""
        return [t.strip() for t in self.relationship_types.split(",") if t.strip()]


settings = Settings()  # type: ignore[call-arg]


@lru_cache
def get_settings() -> Settings:
    return settings
