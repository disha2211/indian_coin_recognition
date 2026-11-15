from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from app.config import (
    PROCESSED_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    SPLIT_DIRNAME,
    TRAIN_INDICES_FILENAME,
    TEST_INDICES_FILENAME,
    SPLIT_METADATA_FILENAME,
    AFE_DIRNAME,
    AFE_FEATURES_FILENAME,
    AFE_LABELS_FILENAME,
)


def _get_class_distribution(
    labels: np.ndarray,
    class_mapping: dict,
) -> dict:
    """Return the number of samples belonging to each class."""

    distribution = {}

    for category, class_index in class_mapping.items():
        distribution[category] = int(
            np.count_nonzero(labels == class_index)
        )

    return distribution


def _validate_existing_split(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    labels: np.ndarray,
) -> bool:
    """Check whether previously saved split indices are still valid."""

    total = len(labels)

    if len(train_indices) + len(test_indices) != total:
        return False

    if len(np.unique(train_indices)) != len(train_indices):
        return False

    if len(np.unique(test_indices)) != len(test_indices):
        return False

    if np.intersect1d(train_indices, test_indices).size:
        return False

    combined = np.concatenate([train_indices, test_indices])

    if not np.array_equal(np.sort(combined), np.arange(total)):
        return False

    # Every class must appear in both sets.
    if len(np.unique(labels[train_indices])) != len(np.unique(labels)):
        return False

    if len(np.unique(labels[test_indices])) != len(np.unique(labels)):
        return False

    return True


def create_or_load_split(dataset_metadata: dict) -> dict:
    """
    Create a reproducible stratified 80/20 split.

    Existing split artifacts are reused when their configuration
    and indices are valid.
    """

    dataset_id = dataset_metadata["dataset_id"]

    processed_dir = PROCESSED_DIR / dataset_id
    afe_dir = processed_dir / AFE_DIRNAME

    features_path = afe_dir / AFE_FEATURES_FILENAME
    labels_path = afe_dir / AFE_LABELS_FILENAME

    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "AFE feature artifacts were not found. "
            "Complete Module 3 before creating the split."
        )

    features = np.load(features_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")

    if len(features) != len(labels):
        raise ValueError(
            "AFE features and labels have different sample counts."
        )

    if len(labels) < 2:
        raise ValueError("Not enough samples to create a train/test split.")

    class_mapping_path = processed_dir / "class_mapping.json"

    if not class_mapping_path.exists():
        raise FileNotFoundError("Class mapping file was not found.")

    class_mapping = json.loads(
        class_mapping_path.read_text(encoding="utf-8")
    )

    if len(np.unique(labels)) != len(class_mapping):
        raise ValueError(
            "The AFE labels do not match the saved class mapping."
        )

    split_dir = processed_dir / SPLIT_DIRNAME
    split_dir.mkdir(parents=True, exist_ok=True)

    train_path = split_dir / TRAIN_INDICES_FILENAME
    test_path = split_dir / TEST_INDICES_FILENAME
    metadata_path = split_dir / SPLIT_METADATA_FILENAME

    # Reuse an existing split when the saved configuration matches.
    if train_path.exists() and test_path.exists() and metadata_path.exists():
        try:
            existing_metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )

            train_indices = np.load(train_path)
            test_indices = np.load(test_path)

            configuration_matches = (
                existing_metadata.get("dataset_id") == dataset_id
                and existing_metadata.get("test_size") == TEST_SIZE
                and existing_metadata.get("random_state") == RANDOM_STATE
                and existing_metadata.get("stratified") is True
                and existing_metadata.get("total_samples") == len(labels)
            )

            if configuration_matches and _validate_existing_split(
                train_indices, test_indices, labels
            ):
                existing_metadata["reused_existing"] = True
                return existing_metadata

        except (OSError, ValueError, json.JSONDecodeError):
            pass

    # Generate a new deterministic, stratified split.
    indices = np.arange(len(labels))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    train_labels = labels[train_indices]
    test_labels = labels[test_indices]

    metadata = {
        "dataset_id": dataset_id,
        "method": "Stratified Train/Test Split",
        "test_size": TEST_SIZE,
        "train_size": 1.0 - TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratified": True,
        "total_samples": len(labels),
        "training_samples": len(train_indices),
        "testing_samples": len(test_indices),
        "class_mapping": class_mapping,
        "class_distribution": {
            "total": _get_class_distribution(labels, class_mapping),
            "training": _get_class_distribution(
                train_labels, class_mapping
            ),
            "testing": _get_class_distribution(
                test_labels, class_mapping
            ),
        },
        "files": {
            "training_indices": str(train_path),
            "testing_indices": str(test_path),
        },
    }

    np.save(train_path, train_indices)
    np.save(test_path, test_indices)

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    del features
    del labels

    metadata["reused_existing"] = False
    return metadata