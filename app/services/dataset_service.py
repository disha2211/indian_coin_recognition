from __future__ import annotations

import json
import os
import shutil
import stat
import uuid
import zipfile
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    EXTRACTED_DIR,
    MAX_ARCHIVE_FILES,
    MAX_EXTRACTED_BYTES,
    UPLOAD_CHUNK_SIZE,
)


def _safe_extract(zip_path: Path, destination: Path) -> None:
    """Extract a ZIP incrementally, with traversal and resource-limit checks."""
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            members = archive.infolist()
            if len(members) > MAX_ARCHIVE_FILES:
                raise ValueError(
                    f"The ZIP contains too many entries ({len(members):,}). "
                    f"The current limit is {MAX_ARCHIVE_FILES:,}."
                )

            declared_total = sum(info.file_size for info in members)
            if declared_total > MAX_EXTRACTED_BYTES:
                raise ValueError(
                    "The extracted dataset would exceed the configured limit "
                    f"of {MAX_EXTRACTED_BYTES / (1024**3):.1f} GiB."
                )

            for info in members:
                # ZIP paths always use forward slashes, including on Windows.
                member_path = Path(info.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError("Invalid ZIP archive: unsafe file path detected.")

                target = (destination / member_path).resolve()
                try:
                    target.relative_to(destination)
                except ValueError:
                    raise ValueError("Invalid ZIP archive: path traversal detected.")

                # Do not create symbolic links from untrusted archives.
                unix_mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(unix_mode):
                    raise ValueError("Invalid ZIP archive: symbolic links are not allowed.")

                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue

                target.parent.mkdir(parents=True, exist_ok=True)
                written = 0
                with archive.open(info, "r") as source, target.open("wb") as output:
                    while True:
                        chunk = source.read(UPLOAD_CHUNK_SIZE)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > info.file_size or written > MAX_EXTRACTED_BYTES:
                            raise ValueError("Invalid ZIP entry size detected during extraction.")
                        output.write(chunk)
    except zipfile.BadZipFile as exc:
        raise ValueError("The uploaded file is not a valid ZIP archive.") from exc


def _find_dataset_root(extracted_dir: Path) -> Path:
    """Find the directory whose immediate subdirectories are image categories.

    Walk bottom-up once and aggregate image counts. This avoids running rglob
    separately for every candidate directory.
    """
    subtree_image_counts: dict[str, int] = {}
    best_path: Path | None = None
    best_score = 0

    for current, dirs, files in os.walk(extracted_dir, topdown=False):
        current_path = Path(current)
        own_images = sum(
            1 for name in files
            if Path(name).suffix.lower().lstrip(".") in ALLOWED_IMAGE_EXTENSIONS
        )
        child_image_dirs = sum(
            1 for dirname in dirs
            if subtree_image_counts.get(str(current_path / dirname), 0) > 0
        )
        total_images = own_images + sum(
            subtree_image_counts.get(str(current_path / dirname), 0) for dirname in dirs
        )
        subtree_image_counts[str(current_path)] = total_images

        # Prefer the directory with the most immediate children containing images.
        if child_image_dirs > best_score:
            best_score = child_image_dirs
            best_path = current_path

    if best_path is None:
        raise ValueError(
            "Dataset structure not recognized. Expected folders containing coin-category image folders."
        )
    return best_path


def _inspect_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def inspect_dataset(zip_path: Path) -> dict:
    dataset_id = uuid.uuid4().hex
    extract_dir = EXTRACTED_DIR / dataset_id
    extract_dir.mkdir(parents=True, exist_ok=False)

    try:
        _safe_extract(zip_path, extract_dir)
        root = _find_dataset_root(extract_dir)

        categories: dict[str, dict] = {}
        invalid_count = 0
        invalid_samples: list[str] = []
        total_valid = 0
        total_candidates = 0

        # os.walk is incremental; no list of every image path is retained.
        for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            category_count = 0
            samples: list[str] = []

            for current, _dirs, filenames in os.walk(category_dir):
                for filename in filenames:
                    file_path = Path(current) / filename
                    if file_path.suffix.lower().lstrip(".") not in ALLOWED_IMAGE_EXTENSIONS:
                        continue

                    total_candidates += 1
                    if _inspect_image(file_path):
                        category_count += 1
                        total_valid += 1
                        if len(samples) < 5:
                            samples.append(str(file_path.relative_to(root)).replace("\\", "/"))
                    else:
                        invalid_count += 1
                        if len(invalid_samples) < 20:
                            invalid_samples.append(str(file_path.relative_to(root)).replace("\\", "/"))

            if category_count:
                categories[category_dir.name] = {
                    "count": category_count,
                    "samples": samples,
                }

        if not categories:
            raise ValueError("No valid coin categories with supported images were found in the uploaded ZIP.")

        metadata = {
            "dataset_id": dataset_id,
            "root": str(root),
            "categories": categories,
            "total_images": total_valid,
            "total_image_candidates": total_candidates,
            "invalid_image_count": invalid_count,
            "invalid_files_sample": invalid_samples,
        }
        metadata_path = extract_dir / "dataset_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata
    except Exception:
        shutil.rmtree(extract_dir, ignore_errors=True)
        raise
