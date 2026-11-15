
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from app.config import MODELS_DIR, PROCESSED_DIR, RANDOM_STATE


# Supported traditional ML models
MODEL_CONFIGS = {
    "naive_bayes": {
        "name": "Multinomial Naive Bayes",
        "estimator": MultinomialNB,
        "params": {
            "alpha": 1.0,
        },
    },
    "decision_tree": {
        "name": "Decision Tree",
        "estimator": DecisionTreeClassifier,
        "params": {
            "random_state": RANDOM_STATE,
        },
    },
    "random_forest": {
        "name": "Random Forest",
        "estimator": RandomForestClassifier,
        "params": {
            "n_estimators": 200,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },
    },
}


def _save_json(data, path):
    """Save JSON safely using a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(".tmp")

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    temp_path.replace(path)


def train_classical_model(dataset_id, model_key, force_retrain=False):
    """
    Train one traditional ML model using saved AFE features
    and the existing train/test split.
    """

    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_key}")

    processed_dir = Path(PROCESSED_DIR) / dataset_id
    afe_dir = processed_dir / "afe"
    split_dir = processed_dir / "split"

    features_path = afe_dir / "features.npy"
    labels_path = afe_dir / "labels.npy"

    train_indices_path = split_dir / "train_indices.npy"
    test_indices_path = split_dir / "test_indices.npy"

    # Verify required files
    required_files = [
        features_path,
        labels_path,
        train_indices_path,
        test_indices_path,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required project file is missing: {path}"
            )

    # Load arrays without unnecessarily copying the full dataset
    features = np.load(features_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")

    train_indices = np.load(train_indices_path)
    test_indices = np.load(test_indices_path)

    if features.ndim != 2:
        raise ValueError("AFE features must be a 2D array.")

    if labels.ndim != 1:
        raise ValueError("Labels must be a 1D array.")

    if len(features) != len(labels):
        raise ValueError("Feature and label counts do not match.")

    if len(train_indices) == 0 or len(test_indices) == 0:
        raise ValueError("Training or testing split is empty.")

    if (
        np.any(train_indices < 0)
        or np.any(train_indices >= len(features))
        or np.any(test_indices < 0)
        or np.any(test_indices >= len(features))
    ):
        raise ValueError("Split contains invalid sample indices.")

    # Load only the training samples into memory
    X_train = np.asarray(
        features[train_indices],
        dtype=np.float32
    )

    y_train = np.asarray(labels[train_indices])

    if not np.isfinite(X_train).all():
        raise ValueError("Training features contain NaN or infinity.")

    if not np.isfinite(y_train).all():
        raise ValueError("Training labels contain invalid values.")

    if len(np.unique(y_train)) < 2:
        raise ValueError("Training data must contain at least two classes.")

    # Multinomial Naive Bayes requires nonnegative features
    if model_key == "naive_bayes" and np.any(X_train < 0):
        raise ValueError(
            "Multinomial Naive Bayes requires nonnegative features."
        )

    model_info = MODEL_CONFIGS[model_key]

    # Keep each dataset's models in its own directory
    model_dir = Path(MODELS_DIR) / dataset_id
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{model_key}.joblib"
    metadata_path = model_dir / f"{model_key}_metadata.json"

    expected_metadata = {
        "dataset_id": dataset_id,
        "model_key": model_key,
        "model_name": model_info["name"],
        "parameters": model_info["params"],
        "feature_count": int(features.shape[1]),
        "training_samples": int(len(train_indices)),
        "testing_samples": int(len(test_indices)),
        "random_state": RANDOM_STATE,
    }

    # Reuse a compatible saved model unless retraining is requested
    if (
        not force_retrain
        and model_path.exists()
        and metadata_path.exists()
    ):
        try:
            with open(metadata_path, "r", encoding="utf-8") as file:
                old_metadata = json.load(file)

            compatible = all(
                old_metadata.get(key) == value
                for key, value in expected_metadata.items()
            )

            if compatible:
                joblib.load(model_path)

                return {
                    "status": "reused",
                    "message": f"{model_info['name']} already exists.",
                    "model_name": model_info["name"],
                    "model_key": model_key,
                    "model_path": str(model_path),
                    "training_samples": len(train_indices),
                    "feature_count": features.shape[1],
                }

        except Exception:
            # If an existing artifact is corrupt, train it again.
            pass

    # Create and train the selected estimator
    model = model_info["estimator"](
        **model_info["params"]
    )

    model.fit(X_train, y_train)

    # Save model atomically
    temp_model_path = model_path.with_suffix(".tmp")
    joblib.dump(model, temp_model_path)
    temp_model_path.replace(model_path)

    metadata = {
        **expected_metadata,
        "status": "trained",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_file": model_path.name,
        "classes": [
            value.item() if isinstance(value, np.generic) else value
            for value in model.classes_
        ],
    }

    _save_json(metadata, metadata_path)

    return {
        "status": "trained",
        "message": f"{model_info['name']} trained successfully.",
        "model_name": model_info["name"],
        "model_key": model_key,
        "model_path": str(model_path),
        "training_samples": len(train_indices),
        "feature_count": features.shape[1],
        "classes": metadata["classes"],
    }
