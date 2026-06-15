#!/usr/bin/env python3
"""Ingest a local folder of documents into OpenGraphKB.

Usage:
    python scripts/ingest_folder.py /path/to/folder
    python scripts/ingest_folder.py /path/to/folder --url http://localhost:8080
    python scripts/ingest_folder.py /path/to/folder --source-type article --company "Acme Corp"
"""

import argparse
import mimetypes
import sys
import time
from pathlib import Path

import httpx

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".md", ".markdown", ".txt", ".text", ".csv", ".json"}


def find_files(folder: Path) -> list[Path]:
    """Recursively find supported files in a folder."""
    files = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return files


def ingest_batch(
    files: list[Path],
    base_url: str,
    source_type: str | None,
    company: str | None,
    batch_size: int,
) -> None:
    """Upload files in batches to the /api/documents/batch endpoint."""
    url = f"{base_url.rstrip('/')}/api/documents/batch"
    total = len(files)
    total_accepted = 0
    total_skipped = 0
    total_failed = 0

    print(f"\n📁 Found {total} files to ingest")
    print(f"📡 Target: {url}")
    if source_type:
        print(f"🏷️  Source type: {source_type}")
    if company:
        print(f"🏢 Company: {company}")
    print(f"📦 Batch size: {batch_size}")
    print("─" * 60)

    start_time = time.time()

    for batch_start in range(0, total, batch_size):
        batch = files[batch_start : batch_start + batch_size]
        batch_num = (batch_start // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size

        print(f"\n📤 Batch {batch_num}/{total_batches} ({len(batch)} files)")

        # Build multipart upload
        upload_files = []
        for file_path in batch:
            content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            upload_files.append(
                ("files", (file_path.name, file_path.read_bytes(), content_type))
            )

        data = {}
        if source_type:
            data["source_type"] = source_type
        if company:
            data["company"] = company

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(url, files=upload_files, data=data)
                response.raise_for_status()
                result = response.json()

                accepted = result.get("accepted", 0)
                skipped = result.get("skipped", 0)
                failed = result.get("failed", 0)
                total_accepted += accepted
                total_skipped += skipped
                total_failed += failed

                # Log per-file results
                for file_result in result.get("results", []):
                    status_icon = "✅" if file_result["accepted"] else ("⏭️" if file_result.get("skipped") else "❌")
                    status_text = (
                        "accepted"
                        if file_result["accepted"]
                        else ("skipped" if file_result.get("skipped") else f"failed: {file_result.get('error', 'unknown')}")
                    )
                    print(f"  {status_icon} {file_result['file_name']} → {file_result['doc_id']} ({status_text})")

        except httpx.HTTPStatusError as exc:
            print(f"  ❌ HTTP error: {exc.response.status_code} — {exc.response.text[:200]}")
            total_failed += len(batch)
        except httpx.ConnectError:
            print(f"  ❌ Connection refused — is the server running at {base_url}?")
            sys.exit(1)
        except Exception as exc:
            print(f"  ❌ Unexpected error: {exc}")
            total_failed += len(batch)

    elapsed = time.time() - start_time
    print("\n" + "─" * 60)
    print(f"✅ Done in {elapsed:.1f}s")
    print(f"   Accepted:  {total_accepted}")
    print(f"   Skipped:   {total_skipped}")
    print(f"   Failed:    {total_failed}")
    print(f"   Total:     {total}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a folder of documents into OpenGraphKB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("folder", type=Path, help="Path to folder containing documents.")
    parser.add_argument("--url", default="http://localhost:8080", help="OpenGraphKB base URL (default: http://localhost:8080).")
    parser.add_argument("--source-type", default=None, help="Source type label applied to all files.")
    parser.add_argument("--company", default=None, help="Company label applied to all files.")
    parser.add_argument("--batch-size", type=int, default=10, help="Files per batch request (default: 10).")
    args = parser.parse_args()

    if not args.folder.is_dir():
        print(f"❌ Not a directory: {args.folder}")
        sys.exit(1)

    files = find_files(args.folder)
    if not files:
        print(f"⚠️  No supported files found in {args.folder}")
        print(f"   Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(0)

    ingest_batch(files, args.url, args.source_type, args.company, args.batch_size)


if __name__ == "__main__":
    main()
