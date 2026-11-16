
from __future__ import annotations

import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import tensorflow as tf

from app.config import (
    IMAGE_SIZE,
    NORMALIZATION_SCALE,
    PROCESSED_DIR,
    MODELS_DIR,
    CLASS_MAPPING_FILENAME,
)

from app.services.afe_service import (
    extract_afe_features,
    apply_clahe,
)


CLASSICAL_MODELS = {
    "naive_bayes": {
        "name": "Multinomial Naive Bayes",
        "filename": "naive_bayes.joblib",
    },
    "decision_tree": {
        "name": "Decision Tree",
        "filename": "decision_tree.joblib",
    },
    "random_forest": {
        "name": "Random Forest",
        "filename": "random_forest.joblib",
    },
}

DCNN_MODEL_KEY = "dcnn"
DCNN_MODEL_NAME = "AFE-Enhanced Deep Convolutional Neural Network"


def _load_class_mapping(dataset_id):
    """Load the saved category-to-label mapping."""

    mapping_path = (
        Path(PROCESSED_DIR)
        / dataset_id
        / CLASS_MAPPING_FILENAME
    )

    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Class mapping not found: {mapping_path}"
        )

    with open(mapping_path, "r", encoding="utf-8") as file:
        mapping = json.load(file)

    label_to_category = {
        int(label): category
        for category, label in mapping.items()
    }

    return label_to_category


def _preprocess_prediction_image(image_path):
    """
    Apply the same preprocessing used during training.

    Output:
        Normalized RGB image of shape (128, 128, 3).
    """

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            "Unable to read the uploaded image."
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    image = (
        image.astype(np.float32)
        / NORMALIZATION_SCALE
    )

    return image


def _predict_classical(
    image,
    dataset_id,
    model_key,
    label_to_category,
):
    """Predict using a saved traditional ML model."""

    model_info = CLASSICAL_MODELS[model_key]

    model_path = (
        Path(MODELS_DIR)
        / dataset_id
        / model_info["filename"]
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found: {model_path}"
        )

    model = joblib.load(model_path)

    # Extract the same 26 AFE features used during training.
    features = extract_afe_features(image)

    if features.shape != (26,):
        raise ValueError(
            f"Expected 26 AFE features, got {features.shape}."
        )

    features = features.reshape(1, -1)

    if not np.isfinite(features).all():
        raise ValueError(
            "Extracted features contain invalid values."
        )

    prediction = int(model.predict(features)[0])

    if prediction not in label_to_category:
        raise ValueError(
            f"Predicted label {prediction} has no category mapping."
        )

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The selected model does not support confidence estimation."
        )

    probabilities = model.predict_proba(features)[0]

    model_classes = [
        int(value) for value in model.classes_
    ]

    if prediction not in model_classes:
        raise ValueError(
            "Predicted class is missing from model metadata."
        )

    predicted_index = model_classes.index(prediction)
    confidence = float(probabilities[predicted_index])

    return {
        "model_key": model_key,
        "model_name": model_info["name"],
        "predicted_label": prediction,
        "predicted_category": label_to_category[prediction],
        "confidence": confidence,
    }


def _predict_dcnn(
    image,
    dataset_id,
    label_to_category,
):
    """Predict using the saved AFE-enhanced DCNN."""

    dcnn_dir = (
        Path(PROCESSED_DIR)
        / dataset_id
        / "dcnn"
    )

    model_path = dcnn_dir / "coin_dcnn.keras"
    labels_path = (
        Path(PROCESSED_DIR)
        / dataset_id
        / "labels.npy"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved DCNN model not found: {model_path}"
        )

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Dataset labels not found: {labels_path}"
        )

    # DCNN training used sorted unique numeric labels
    # encoded as consecutive output indices.
    labels = np.load(
        labels_path,
        mmap_mode="r",
    )

    original_classes = np.unique(labels)

    # Apply the same CLAHE grayscale enhancement as DCNN training.
    enhanced = apply_clahe(image)

    dcnn_input = enhanced.reshape(
        1,
        IMAGE_SIZE[1],
        IMAGE_SIZE[0],
        1,
    ).astype(np.float32)

    model = tf.keras.models.load_model(
        model_path
    )

    probabilities = model.predict(
        dcnn_input,
        verbose=0,
    )[0]

    predicted_index = int(np.argmax(probabilities))

    if predicted_index >= len(original_classes):
        raise ValueError(
            "DCNN predicted an invalid class index."
        )

    predicted_label = int(
        original_classes[predicted_index]
    )

    if predicted_label not in label_to_category:
        raise ValueError(
            "DCNN prediction has no category mapping."
        )

    return {
        "model_key": DCNN_MODEL_KEY,
        "model_name": DCNN_MODEL_NAME,
        "predicted_label": predicted_label,
        "predicted_category": label_to_category[predicted_label],
        "confidence": float(probabilities[predicted_index]),
    }


def predict_coin(image_path, dataset_id, model_key="random_forest"):
    """
    Predict an Indian coin denomination using a saved model.

    Supported models:
        random_forest
        decision_tree
        naive_bayes
        dcnn
    """

    image_path = Path(image_path)

    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(
            "Uploaded image could not be found."
        )

    if image_path.suffix.lower() not in {
        ".jpg",
        ".jpeg",
        ".png",
    }:
        raise ValueError(
            "Only JPG, JPEG and PNG images are supported."
        )

    supported_models = {
        *CLASSICAL_MODELS.keys(),
        DCNN_MODEL_KEY,
    }

    if model_key not in supported_models:
        raise ValueError(
            f"Unsupported prediction model: {model_key}"
        )

    label_to_category = _load_class_mapping(dataset_id)

    image = _preprocess_prediction_image(
        image_path
    )

    if model_key == DCNN_MODEL_KEY:
        result = _predict_dcnn(
            image,
            dataset_id,
            label_to_category,
        )
    else:
        result = _predict_classical(
            image,
            dataset_id,
            model_key,
            label_to_category,
        )

    result["dataset_id"] = dataset_id
    result["confidence_percentage"] = round(
        result["confidence"] * 100,
        2,
    )

    return result
