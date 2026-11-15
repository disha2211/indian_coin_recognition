from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np

from app.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    IMAGE_SIZE,
    NORMALIZATION_SCALE,
    PREPROCESS_DTYPE,
    PROCESSED_DIR,
    PROCESSED_IMAGES_FILENAME,
    PROCESSED_LABELS_FILENAME,
    PREPROCESS_METADATA_FILENAME,
    CLASS_MAPPING_FILENAME,
)


def _is_supported_image(path: Path) -> bool:
    return path.suffix.lower().lstrip(".") in ALLOWED_IMAGE_EXTENSIONS


def _iter_category_images(category_dir: Path):
    """
    Yield supported image paths one at a time.

    Images are not accumulated in memory.
    """
    for current, _dirs, filenames in os.walk(category_dir):
        for filename in sorted(filenames):
            path = Path(current) / filename

            if _is_supported_image(path):
                yield path


def _preprocess_image(image_path: Path) -> np.ndarray:
    """
    Read and preprocess one image.

    Output:
        float32 RGB image with shape (128, 128, 3)
        and values normalized to [0, 1].
    """

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    # OpenCV loads images as BGR.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to the project's fixed image size.
    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    # Convert to float32 and normalize pixel values.
    image = image.astype(np.float32) / NORMALIZATION_SCALE

    return image


def preprocess_dataset(dataset_metadata: dict) -> dict:
    """
    Preprocess the currently extracted dataset using disk-backed
    NumPy arrays.

    The complete image dataset is never loaded into RAM.
    """

    dataset_id = dataset_metadata["dataset_id"]
    root = Path(dataset_metadata["root"])

    total_images = int(dataset_metadata["total_images"])

    if total_images <= 0:
        raise ValueError("There are no valid images available for preprocessing.")

    categories = sorted(dataset_metadata["categories"].keys())

    if len(categories) < 2:
        raise ValueError(
            "At least two coin categories are required for classification."
        )

    # Stable category -> integer label mapping.
    class_mapping = {
        category: index
        for index, category in enumerate(categories)
    }

    # Create a dedicated directory for this preprocessing run.
    processed_dir = PROCESSED_DIR / dataset_id
    processed_dir.mkdir(parents=True, exist_ok=True)

    images_path = processed_dir / PROCESSED_IMAGES_FILENAME
    labels_path = processed_dir / PROCESSED_LABELS_FILENAME
    metadata_path = processed_dir / PREPROCESS_METADATA_FILENAME
    class_mapping_path = processed_dir / CLASS_MAPPING_FILENAME

    # Disk-backed arrays.
    X = np.lib.format.open_memmap(
        images_path,
        mode="w+",
        dtype=PREPROCESS_DTYPE,
        shape=(
            total_images,
            IMAGE_SIZE[1],
            IMAGE_SIZE[0],
            3,
        ),
    )

    y = np.lib.format.open_memmap(
        labels_path,
        mode="w+",
        dtype=np.int64,
        shape=(total_images,),
    )

    processed_count = 0
    failed_count = 0
    failed_samples: list[str] = []

    # Process one image at a time.
    for category in categories:
        category_dir = root / category

        if not category_dir.exists() or not category_dir.is_dir():
            continue

        label = class_mapping[category]

        for image_path in _iter_category_images(category_dir):

            try:
                processed_image = _preprocess_image(image_path)

                if processed_count >= total_images:
                    raise RuntimeError(
                        "More images were discovered during preprocessing "
                        "than were recorded during dataset inspection."
                    )

                X[processed_count] = processed_image
                y[processed_count] = label

                processed_count += 1

            except (ValueError, OSError, cv2.error) as exc:
                failed_count += 1

                if len(failed_samples) < 20:
                    failed_samples.append(
                        {
                            "file": str(image_path.relative_to(root)).replace("\\", "/"),
                            "error": str(exc),
                        }
                    )

    # Make sure the metadata count and actual processed count agree.
    if processed_count != total_images:
        del X
        del y

        # Remove incomplete outputs.
        images_path.unlink(missing_ok=True)
        labels_path.unlink(missing_ok=True)

        raise ValueError(
            "Preprocessing did not process the expected number of images. "
            f"Expected {total_images}, processed {processed_count}. "
            "The dataset may have changed after inspection."
        )

    # Flush memory-mapped arrays to disk.
    X.flush()
    y.flush()

    # Release memory-map handles.
    del X
    del y

    preprocessing_metadata = {
        "dataset_id": dataset_id,
        "image_size": list(IMAGE_SIZE),
        "color_format": "RGB",
        "dtype": PREPROCESS_DTYPE,
        "normalization": {
            "method": "pixel_value / 255.0",
            "input_range": [0, 255],
            "output_range": [0.0, 1.0],
        },
        "processed_images": processed_count,
        "failed_images": failed_count,
        "class_count": len(categories),
        "class_mapping": class_mapping,
        "files": {
            "images": str(images_path),
            "labels": str(labels_path),
        },
        "failed_samples": failed_samples,
    }

    metadata_path.write_text(
        json.dumps(preprocessing_metadata, indent=2),
        encoding="utf-8",
    )

    class_mapping_path.write_text(
        json.dumps(class_mapping, indent=2),
        encoding="utf-8",
    )

    return preprocessing_metadata