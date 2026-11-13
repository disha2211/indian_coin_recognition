from __future__ import annotations

import json
import shutil
import uuid
import zipfile
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.config import ALLOWED_IMAGE_EXTENSIONS, EXTRACTED_DIR


def _safe_extract(zip_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not str(target).startswith(str(destination)):
                raise ValueError("Invalid ZIP archive: path traversal detected.")
        archive.extractall(destination)


def _find_dataset_root(extracted_dir: Path) -> Path:
    """Find the directory containing category directories.

    The expected dataset convention is:
        dataset_root/<category>/<images...>
    A single extra wrapper folder is supported.
    """
    candidates = [extracted_dir]
    candidates.extend(p for p in extracted_dir.rglob("*") if p.is_dir())

    best = None
    best_count = 0
    for directory in candidates:
        child_dirs = [p for p in directory.iterdir() if p.is_dir()]
        if not child_dirs:
            continue
        category_like = 0
        for child in child_dirs:
            if any(f.suffix.lower().lstrip(".") in ALLOWED_IMAGE_EXTENSIONS for f in child.rglob("*")):
                category_like += 1
        if category_like > best_count:
            best = directory
            best_count = category_like

    if best is None:
        raise ValueError(
            "Dataset structure not recognized. Expected folders containing coin-category image folders."
        )
    return best


def _inspect_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def inspect_dataset(zip_path: Path) -> dict:
    dataset_id = uuid.uuid4().hex
    extract_dir = EXTRACTED_DIR / dataset_id
    extract_dir.mkdir(parents=True, exist_ok=False)

    try:
        _safe_extract(zip_path, extract_dir)
        root = _find_dataset_root(extract_dir)

        categories = {}
        invalid_files = []
        total_valid = 0

        for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            files = []
            for file_path in sorted(category_dir.rglob("*")):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower().lstrip(".") not in ALLOWED_IMAGE_EXTENSIONS:
                    continue
                if _inspect_image(file_path):
                    files.append(file_path)
                    total_valid += 1
                else:
                    invalid_files.append(str(file_path.relative_to(root)))
            if files:
                categories[category_dir.name] = {
                    "count": len(files),
                    "samples": [str(p.relative_to(root)).replace("\\", "/") for p in files[:5]],
                }

        if not categories:
            raise ValueError(
                "No valid coin categories with supported images were found in the uploaded ZIP."
            )

        metadata = {
            "dataset_id": dataset_id,
            "root": str(root),
            "categories": categories,
            "total_images": total_valid,
            "invalid_files": invalid_files,
        }
        (extract_dir / "dataset_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return metadata
    except Exception:
        shutil.rmtree(extract_dir, ignore_errors=True)
        raise
