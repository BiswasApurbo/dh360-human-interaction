#!/usr/bin/env python3
"""Model acquisition and verification script.

Downloads required pretrained computer vision models and test fixtures with
cryptographic checksum validation.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_models")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "weights"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"


@dataclass(frozen=True)
class ResourceEntry:
    """Metadata for a downloadable pretrained model or fixture."""

    name: str
    filename: str
    url: str
    sha256: str
    default_dir: Path
    description: str


REGISTRY: dict[str, ResourceEntry] = {
    "yunet": ResourceEntry(
        name="YuNet (Face Detection)",
        filename="face_detection_yunet_2023mar.onnx",
        url="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        default_dir=MODELS_DIR,
        description="Lightweight YuNet face detector from OpenCV Model Zoo (2023-Mar version)",
    ),
    "sample_face": ResourceEntry(
        name="OpenCV Standard Sample Face (Fixture)",
        filename="sample_face.jpg",
        url="https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg",
        sha256="7de7ed51a1594fff247f4cae2301eceacf5313d6011e37b4a4c8733f7bb72c07",
        default_dir=FIXTURES_DIR,
        description="Standard reference portrait fixture from OpenCV repository (BSD-3-Clause)",
    ),
}


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_resource(
    entry: ResourceEntry, target_dir: Path | None = None, force: bool = False
) -> Path:
    """Download and verify a resource entry."""
    out_dir = target_dir or entry.default_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / entry.filename

    if destination.exists() and not force:
        logger.info("Found existing file: %s", destination)
        current_hash = compute_sha256(destination)
        if current_hash == entry.sha256:
            logger.info("SHA-256 checksum verified for %s", entry.name)
            return destination
        logger.warning(
            "Checksum mismatch for %s (expected %s, got %s). Re-downloading...",
            entry.filename,
            entry.sha256,
            current_hash,
        )

    logger.info("Downloading %s from %s...", entry.name, entry.url)
    req = urllib.request.Request(entry.url, headers={"User-Agent": "dh360-model-downloader/0.1.0"})

    try:
        with (
            urllib.request.urlopen(req, timeout=30) as response,
            destination.open("wb") as out_file,
        ):
            while chunk := response.read(65536):
                out_file.write(chunk)
    except Exception as exc:
        if destination.exists():
            destination.unlink(missing_ok=True)
        msg = f"Failed to download resource '{entry.name}' from {entry.url}: {exc}"
        raise RuntimeError(msg) from exc

    downloaded_hash = compute_sha256(destination)
    if downloaded_hash != entry.sha256:
        destination.unlink(missing_ok=True)
        raise ValueError(
            f"Checksum verification failed for {entry.filename}!\n"
            f"Expected: {entry.sha256}\n"
            f"Actual:   {downloaded_hash}"
        )

    logger.info("Successfully downloaded and verified %s -> %s", entry.name, destination)
    return destination


def download_all_resources(force: bool = False) -> dict[str, Path]:
    """Download all registered models and test fixtures."""
    results = {}
    for key, entry in REGISTRY.items():
        results[key] = download_resource(entry, force=force)
    return results


def main() -> int:
    """CLI entrypoint for resource downloader."""
    parser = argparse.ArgumentParser(
        description="Download pretrained model weights and test fixtures for dh360"
    )
    parser.add_argument(
        "--resource",
        choices=list(REGISTRY.keys()) + ["all"],
        default="all",
        help="Specific resource to download (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file already exists with matching checksum",
    )
    args = parser.parse_args()

    try:
        if args.resource == "all":
            download_all_resources(force=args.force)
        else:
            download_resource(REGISTRY[args.resource], force=args.force)
        logger.info("All requested resources are ready.")
        return 0
    except Exception as e:
        logger.error("Error during download: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
