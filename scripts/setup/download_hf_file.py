"""Download a pinned public Hugging Face file in resumable byte ranges."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests


CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_metadata(url: str, session: requests.Session) -> tuple[int, str]:
    response = session.head(url, allow_redirects=False, timeout=(15, 60))
    response.raise_for_status()
    size = response.headers.get("X-Linked-Size") or response.headers.get("Content-Length")
    checksum = response.headers.get("X-Linked-ETag", "").strip('"')
    if not size or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise RuntimeError("Hub did not provide a size and SHA-256 checksum")
    return int(size), checksum


def download_file(
    url: str,
    target: Path,
    chunk_bytes: int,
    max_retries: int,
    partial_path: Path | None = None,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = partial_path or target.with_name(target.name + ".part")
    with requests.Session() as session:
        total_size, expected_hash = resolve_metadata(url, session)
        if target.exists():
            if target.stat().st_size == total_size and file_sha256(target) == expected_hash:
                print(f"Verified existing file: {target}", flush=True)
                return
            raise RuntimeError(f"Existing target does not match the Hub file: {target}")
        if partial.exists() and partial.stat().st_size > total_size:
            raise RuntimeError(f"Partial file is larger than the Hub file: {partial}")

        failures = 0
        while True:
            start = partial.stat().st_size if partial.exists() else 0
            if start == total_size:
                break
            end = min(start + chunk_bytes, total_size) - 1
            try:
                with session.get(
                    url,
                    headers={"Range": f"bytes={start}-{end}"},
                    stream=True,
                    timeout=(15, 60),
                ) as response:
                    if response.status_code != 206:
                        raise RuntimeError(f"Expected HTTP 206; received {response.status_code}")
                    match = CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
                    if not match or tuple(map(int, match.groups())) != (start, end, total_size):
                        raise RuntimeError("Unexpected Content-Range from the download server")
                    with partial.open("ab") as output:
                        for block in response.iter_content(chunk_size=1024 * 1024):
                            if block:
                                output.write(block)
                failures = 0
                complete = partial.stat().st_size
                print(f"{target.name}: {complete:,}/{total_size:,} bytes", flush=True)
            except (requests.RequestException, OSError) as error:
                failures += 1
                if failures > max_retries:
                    raise RuntimeError(f"Download stalled after {max_retries} retries") from error
                print(f"Retry {failures}/{max_retries}: {error}", flush=True)
                time.sleep(min(2**failures, 15))

    actual_hash = file_sha256(partial)
    if actual_hash != expected_hash:
        raise RuntimeError(f"SHA-256 mismatch for {partial}: {actual_hash} != {expected_hash}")
    os.replace(partial, target)
    print(f"Verified SHA-256 and saved: {target}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository")
    parser.add_argument("revision")
    parser.add_argument("filename")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--repo-type", choices=("model", "dataset"), default="model")
    parser.add_argument("--chunk-mb", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--partial-path", type=Path)
    args = parser.parse_args()
    if args.chunk_mb <= 0 or args.max_retries < 0:
        parser.error("--chunk-mb must be positive and --max-retries must be non-negative")
    prefix = "datasets/" if args.repo_type == "dataset" else ""
    filename = quote(args.filename, safe="/")
    url = f"https://huggingface.co/{prefix}{args.repository}/resolve/{args.revision}/{filename}"
    download_file(
        url,
        args.output_directory / args.filename,
        args.chunk_mb * 1024 * 1024,
        args.max_retries,
        args.partial_path,
    )


if __name__ == "__main__":
    main()
