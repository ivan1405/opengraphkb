import json
from pathlib import Path

from core.logging import get_logger

log = get_logger("storage")


class StorageService:
    """Local filesystem document storage.

    Stores raw files, normalized text, and JSON metadata under a
    structured directory layout::

        {base_path}/
        ├── documents/{doc_id}/{filename}
        ├── normalized/{doc_id}/content.md
        └── metadata/{doc_id}.json
    """

    def __init__(self, base_path: str) -> None:
        self._base = Path(base_path)
        self._documents_dir = self._base / "documents"
        self._normalized_dir = self._base / "normalized"
        self._metadata_dir = self._base / "metadata"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        for directory in (self._documents_dir, self._normalized_dir, self._metadata_dir):
            directory.mkdir(parents=True, exist_ok=True)

    # ── Raw Documents ─────────────────────────────────────────────────────

    async def upload_document(
        self, doc_id: str, filename: str, content: bytes, content_type: str
    ) -> str:
        doc_dir = self._documents_dir / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        file_path = doc_dir / filename
        file_path.write_bytes(content)
        log.info("document_stored", doc_id=doc_id, filename=filename, bytes=len(content))
        return str(file_path)

    async def download_document(self, doc_id: str, filename: str) -> bytes:
        file_path = self._documents_dir / doc_id / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Document file not found: {file_path}")
        return file_path.read_bytes()

    async def delete_document(self, doc_id: str) -> None:
        doc_dir = self._documents_dir / doc_id
        if doc_dir.exists():
            import shutil

            shutil.rmtree(doc_dir)
            log.info("document_files_deleted", doc_id=doc_id)

    # ── Normalized Text ───────────────────────────────────────────────────

    async def upload_normalized(self, doc_id: str, text: str) -> None:
        norm_dir = self._normalized_dir / doc_id
        norm_dir.mkdir(parents=True, exist_ok=True)
        (norm_dir / "content.md").write_text(text, encoding="utf-8")

    async def download_normalized(self, doc_id: str) -> str | None:
        file_path = self._normalized_dir / doc_id / "content.md"
        if not file_path.exists():
            return None
        return file_path.read_text(encoding="utf-8")

    async def delete_normalized(self, doc_id: str) -> None:
        norm_dir = self._normalized_dir / doc_id
        if norm_dir.exists():
            import shutil

            shutil.rmtree(norm_dir)

    # ── Metadata ──────────────────────────────────────────────────────────

    async def save_metadata(self, doc_id: str, meta: dict) -> None:
        file_path = self._metadata_dir / f"{doc_id}.json"
        file_path.write_text(json.dumps(meta, default=str, indent=2), encoding="utf-8")

    async def get_metadata(self, doc_id: str) -> dict | None:
        file_path = self._metadata_dir / f"{doc_id}.json"
        if not file_path.exists():
            return None
        return json.loads(file_path.read_text(encoding="utf-8"))

    async def list_metadata(self) -> list[dict]:
        items: list[dict] = []
        if not self._metadata_dir.exists():
            return items
        for file_path in sorted(self._metadata_dir.glob("*.json")):
            try:
                meta = json.loads(file_path.read_text(encoding="utf-8"))
                items.append(meta)
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("metadata_read_failed", path=str(file_path), error=str(exc))
        return items

    async def delete_metadata(self, doc_id: str) -> None:
        file_path = self._metadata_dir / f"{doc_id}.json"
        if file_path.exists():
            file_path.unlink()
