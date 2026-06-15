class OpenGraphKBError(Exception):
    """Base exception for all OpenGraphKB errors."""


class DocumentNotFoundError(OpenGraphKBError):
    """Raised when a document is not found in storage."""

    def __init__(self, doc_id: str) -> None:
        self.doc_id = doc_id
        super().__init__(f"Document '{doc_id}' not found")


class DocumentAlreadyExistsError(OpenGraphKBError):
    """Raised when a document with the given ID already exists."""

    def __init__(self, doc_id: str) -> None:
        self.doc_id = doc_id
        super().__init__(f"Document '{doc_id}' already exists")


class ParserNotFoundError(OpenGraphKBError):
    """Raised when no parser is available for a given content type."""

    def __init__(self, content_type: str, filename: str) -> None:
        self.content_type = content_type
        self.filename = filename
        super().__init__(
            f"No parser available for content_type='{content_type}', filename='{filename}'"
        )


class PipelineError(OpenGraphKBError):
    """Raised when the ingestion pipeline encounters an unrecoverable error."""


class ExtractionError(OpenGraphKBError):
    """Raised when LLM entity or relationship extraction fails."""


class GraphServiceError(OpenGraphKBError):
    """Raised when a Neo4j operation fails."""


class VectorServiceError(OpenGraphKBError):
    """Raised when a Qdrant operation fails."""
