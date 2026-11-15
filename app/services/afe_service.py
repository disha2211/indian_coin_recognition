from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np

from app.config import (
    AFE_CLAHE_CLIP_LIMIT,
    AFE_CLAHE_TILE_GRID_SIZE,
    AFE_CONFIG_FILENAME,
    AFE_DIRNAME,
    AFE_FEATURE_DTYPE,
    AFE_FEATURES_FILENAME,
    AFE_LABELS_FILENAME,
    AFE_LBP_POINTS,
    AFE_LBP_RADIUS,
    AFE_METADATA_FILENAME,
    AFE_SOBEL_KERNEL_SIZE,
    PROCESSED_DIR,
)

def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Enhance local contrast using CLAHE.

    Input:
        RGB image with pixel values in [0, 1].

    Output:
        CLAHE-enhanced grayscale image with values in [0, 1].
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("AFE expects an RGB image.")

    # Convert normalized RGB [0,1] -> uint8 [0,255]
    rgb_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    # Convert RGB to grayscale
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=AFE_CLAHE_CLIP_LIMIT,
        tileGridSize=AFE_CLAHE_TILE_GRID_SIZE,
    )

    enhanced = clahe.apply(gray)

    return enhanced.astype(np.float32) / 255.0


def extract_sobel_features(
    enhanced_image: np.ndarray,
) -> np.ndarray:
    """
    Extract statistical gradient features using Sobel operators.
    """
    image_uint8 = np.clip(
        enhanced_image * 255.0,
        0,
        255,
    ).astype(np.uint8)

    sobel_x = cv2.Sobel(
        image_uint8,
        cv2.CV_32F,
        1,
        0,
        ksize=AFE_SOBEL_KERNEL_SIZE,
    )

    sobel_y = cv2.Sobel(
        image_uint8,
        cv2.CV_32F,
        0,
        1,
        ksize=AFE_SOBEL_KERNEL_SIZE,
    )

    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)

    # Normalize gradient values for numerical stability.
    max_value = float(gradient_magnitude.max())

    if max_value > 0:
        gradient_magnitude /= max_value

    features = np.array(
        [
            float(gradient_magnitude.mean()),
            float(gradient_magnitude.std()),
            float(gradient_magnitude.min()),
            float(gradient_magnitude.max()),
            float(np.percentile(gradient_magnitude, 25)),
            float(np.percentile(gradient_magnitude, 50)),
            float(np.percentile(gradient_magnitude, 75)),
            float(np.percentile(gradient_magnitude, 90)),
        ],
        dtype=np.float32,
    )

    return features


def extract_lbp_histogram(
    enhanced_image: np.ndarray,
) -> np.ndarray:
    """
    Extract a normalized Local Binary Pattern histogram.

    Uses:
        P = 8 neighboring points
        R = 1 pixel radius

    Uniform LBP produces P + 2 histogram bins.
    """
    image = enhanced_image.astype(np.float32)

    radius = AFE_LBP_RADIUS
    points = AFE_LBP_POINTS

    if radius != 1 or points != 8:
        raise ValueError(
            "The current LBP implementation supports "
            "8 points with radius 1."
        )

    # Pad the image so every pixel has eight neighbors.
    padded = np.pad(
        image,
        pad_width=radius,
        mode="edge",
    )

    center = padded[
        radius:-radius,
        radius:-radius,
    ]

    # Clockwise 8-neighbor offsets.
    neighbors = [
        padded[0:-2, 0:-2],
        padded[0:-2, 1:-1],
        padded[0:-2, 2:],
        padded[1:-1, 2:],
        padded[2:, 2:],
        padded[2:, 1:-1],
        padded[2:, 0:-2],
        padded[1:-1, 0:-2],
    ]

    binary_pattern = np.zeros_like(
        center,
        dtype=np.uint8,
    )

    for bit, neighbor in enumerate(neighbors):
        binary_pattern |= (
            (neighbor >= center).astype(np.uint8) << bit
        )

    # Convert ordinary 8-bit LBP into uniform-LBP categories.
    uniform_pattern = np.zeros_like(
        binary_pattern,
        dtype=np.uint8,
    )

    for value in range(256):
        bits = format(value, "08b")

        transitions = sum(
            bits[i] != bits[(i + 1) % 8]
            for i in range(8)
        )

        if transitions <= 2:
            ones = bits.count("1")
            uniform_pattern[binary_pattern == value] = ones
        else:
            uniform_pattern[binary_pattern == value] = 9

    histogram = np.bincount(
        uniform_pattern.ravel(),
        minlength=10,
    ).astype(np.float32)

    total = histogram.sum()

    if total > 0:
        histogram /= total

    return histogram


def extract_afe_features(image: np.ndarray) -> np.ndarray:
    """
    Complete Adaptive Feature Enhancement pipeline.

    Pipeline:
        RGB image
            ↓
        CLAHE local contrast enhancement
            ↓
        Sobel structural features
            ↓
        Uniform LBP texture histogram
            ↓
        Combined numerical feature vector
    """
    enhanced = apply_clahe(image)

    sobel_features = extract_sobel_features(
        enhanced
    )

    lbp_features = extract_lbp_histogram(
        enhanced
    )

    # Basic statistics of the enhanced image.
    intensity_features = np.array(
        [
            float(enhanced.mean()),
            float(enhanced.std()),
            float(enhanced.min()),
            float(enhanced.max()),
            float(np.percentile(enhanced, 25)),
            float(np.percentile(enhanced, 50)),
            float(np.percentile(enhanced, 75)),
            float(np.percentile(enhanced, 90)),
        ],
        dtype=np.float32,
    )

    features = np.concatenate(
        [
            intensity_features,
            sobel_features,
            lbp_features,
        ]
    ).astype(np.float32)

    return features

def extract_afe_dataset(dataset_metadata: dict) -> dict:
    """
    Apply AFE to the complete preprocessed dataset.

    Images are read through NumPy memory mapping so the complete
    image dataset does not have to be loaded into RAM.
    """

    dataset_id = dataset_metadata["dataset_id"]
    total_images = int(dataset_metadata["total_images"])

    processed_dir = (
        Path(dataset_metadata["root"])
        .resolve()
        .parents[1]
    )

    # The processed directory is derived from the dataset ID rather
    # than relying on the extracted dataset location.
    processed_dir = PROCESSED_DIR / dataset_id

    afe_dir = processed_dir / AFE_DIRNAME
    afe_dir.mkdir(parents=True, exist_ok=True)

    images_path = processed_dir / "images.npy"
    labels_path = processed_dir / "labels.npy"

    features_path = afe_dir / AFE_FEATURES_FILENAME
    afe_labels_path = afe_dir / AFE_LABELS_FILENAME
    metadata_path = afe_dir / AFE_METADATA_FILENAME
    config_path = afe_dir / AFE_CONFIG_FILENAME

    if not images_path.exists():
        raise FileNotFoundError(
            "Preprocessed images were not found. "
            "Complete Module 2 before running AFE."
        )

    if not labels_path.exists():
        raise FileNotFoundError(
            "Preprocessed labels were not found. "
            "Complete Module 2 before running AFE."
        )

    # Memory-map the preprocessing output.
    images = np.load(
        images_path,
        mmap_mode="r",
    )

    labels = np.load(
        labels_path,
        mmap_mode="r",
    )

    if len(images) != total_images:
        raise ValueError(
            "The number of preprocessed images does not match "
            f"dataset metadata. Expected {total_images}, "
            f"found {len(images)}."
        )

    if len(labels) != total_images:
        raise ValueError(
            "The number of labels does not match "
            f"dataset metadata. Expected {total_images}, "
            f"found {len(labels)}."
        )

    # AFE currently produces 26 features per image.
    feature_count = 26

    features = np.lib.format.open_memmap(
        features_path,
        mode="w+",
        dtype=AFE_FEATURE_DTYPE,
        shape=(total_images, feature_count),
    )

    afe_labels = np.lib.format.open_memmap(
        afe_labels_path,
        mode="w+",
        dtype=np.int64,
        shape=(total_images,),
    )

    processed_count = 0
    failed_count = 0
    failed_samples: list[dict] = []

    try:
        for index in range(total_images):
            try:
                image = images[index]

                extracted_features = extract_afe_features(
                    image
                )

                if extracted_features.shape != (feature_count,):
                    raise ValueError(
                        "Unexpected AFE feature dimension. "
                        f"Expected {(feature_count,)}, "
                        f"received {extracted_features.shape}."
                    )

                features[index] = extracted_features
                afe_labels[index] = labels[index]

                processed_count += 1

            except (ValueError, OSError, cv2.error) as exc:
                failed_count += 1

                if len(failed_samples) < 20:
                    failed_samples.append(
                        {
                            "index": index,
                            "error": str(exc),
                        }
                    )

        if processed_count != total_images:
            raise ValueError(
                "AFE processing did not complete successfully. "
                f"Expected {total_images} images, "
                f"processed {processed_count}, "
                f"failed {failed_count}."
            )

        features.flush()
        afe_labels.flush()

    except Exception:
        del features
        del afe_labels

        features_path.unlink(missing_ok=True)
        afe_labels_path.unlink(missing_ok=True)

        raise

    finally:
        del images
        del labels

    afe_config = {
        "method": "Adaptive Feature Enhancement",
        "version": "1.0",
        "input": {
            "color_format": "RGB",
            "image_size": [128, 128],
            "normalized_range": [0.0, 1.0],
        },
        "stages": [
            {
                "stage": 1,
                "name": "Local Contrast Enhancement",
                "method": "CLAHE",
                "parameters": {
                    "clip_limit": AFE_CLAHE_CLIP_LIMIT,
                    "tile_grid_size": list(
                        AFE_CLAHE_TILE_GRID_SIZE
                    ),
                },
            },
            {
                "stage": 2,
                "name": "Structural Feature Extraction",
                "method": "Sobel Gradient",
                "parameters": {
                    "kernel_size": AFE_SOBEL_KERNEL_SIZE,
                },
            },
            {
                "stage": 3,
                "name": "Texture Feature Extraction",
                "method": "Uniform Local Binary Pattern",
                "parameters": {
                    "points": AFE_LBP_POINTS,
                    "radius": AFE_LBP_RADIUS,
                    "histogram_bins": 10,
                },
            },
        ],
        "feature_groups": {
            "enhanced_intensity_statistics": 8,
            "sobel_gradient_statistics": 8,
            "lbp_histogram": 10,
        },
        "total_features": feature_count,
    }

    afe_metadata = {
        "dataset_id": dataset_id,
        "method": "Adaptive Feature Enhancement",
        "processed_images": processed_count,
        "failed_images": failed_count,
        "feature_count": feature_count,
        "feature_dtype": AFE_FEATURE_DTYPE,
        "files": {
            "features": str(features_path),
            "labels": str(afe_labels_path),
            "config": str(config_path),
        },
        "failed_samples": failed_samples,
    }

    config_path.write_text(
        json.dumps(
            afe_config,
            indent=2,
        ),
        encoding="utf-8",
    )

    metadata_path.write_text(
        json.dumps(
            afe_metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    return afe_metadata