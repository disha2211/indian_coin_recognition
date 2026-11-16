
import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from app.config import PROCESSED_DIR, MODELS_DIR


CLASSICAL_MODELS = {
    "naive_bayes": "Multinomial Naive Bayes",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
}

DCNN_MODEL_KEY = "dcnn"
DCNN_MODEL_NAME = "AFE-Enhanced Deep Convolutional Neural Network"


def _save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(".tmp")

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    temp_path.replace(path)


def _evaluate_predictions(y_true, y_pred, classes, class_names):
    """Calculate classification metrics and confusion matrix."""

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),

        "precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                labels=classes,
                average="macro",
                zero_division=0,
            )
        ),

        "recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                labels=classes,
                average="macro",
                zero_division=0,
            )
        ),

        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                labels=classes,
                average="macro",
                zero_division=0,
            )
        ),

        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=classes,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),

        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=classes,
        ).tolist(),
    }


def _predict_dcnn(model, images, test_indices, batch_size=16):
    """Predict using batches to limit memory consumption."""

    predictions = []

    for start in range(0, len(test_indices), batch_size):
        batch_indices = test_indices[
            start:start + batch_size
        ]

        batch_images = np.asarray(
            images[batch_indices],
            dtype=np.float32,
        )

        probabilities = model.predict(
            batch_images,
            verbose=0,
        )

        predictions.extend(
            np.argmax(probabilities, axis=1).tolist()
        )

    return np.asarray(predictions, dtype=np.int64)



def _save_comparison_plot(comparison, evaluation_dir):
    """Save a grouped metric comparison chart for all evaluated models."""

    model_names = [item["model_name"] for item in comparison["models"]]
    metric_specs = [
        ("accuracy", "Accuracy"),
        ("precision_macro", "Macro Precision"),
        ("recall_macro", "Macro Recall"),
        ("f1_macro", "Macro F1-score"),
    ]

    x = np.arange(len(model_names))
    width = 0.19

    fig, ax = plt.subplots(figsize=(12, 6))

    for index, (metric_key, label) in enumerate(metric_specs):
        values = [
            item[metric_key]
            for item in comparison["models"]
        ]
        offsets = x + (index - 1.5) * width
        bars = ax.bar(offsets, values, width, label=label)
        ax.bar_label(
            bars,
            labels=[f"{value:.2f}" for value in values],
            padding=2,
            fontsize=8,
        )

    ax.set_title("Four-Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=12, ha="right")
    ax.legend(loc="upper left", ncols=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    plot_path = evaluation_dir / "comparison_metrics.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return plot_path.name


def evaluate_all_models(dataset_id, dataset_metadata):
    """
    Evaluate all three traditional ML models and the DCNN
    on the same saved test indices.
    """

    processed_dir = Path(PROCESSED_DIR) / dataset_id
    models_dir = Path(MODELS_DIR) / dataset_id

    afe_dir = processed_dir / "afe"
    dcnn_dir = processed_dir / "dcnn"
    split_dir = processed_dir / "split"

    labels_path = processed_dir / "labels.npy"
    features_path = afe_dir / "features.npy"
    dcnn_images_path = dcnn_dir / "afe_images.npy"

    test_indices_path = split_dir / "test_indices.npy"

    required_files = [
        labels_path,
        features_path,
        dcnn_images_path,
        test_indices_path,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Evaluation input missing: {path}"
            )

    labels = np.load(labels_path, mmap_mode="r")
    features = np.load(features_path, mmap_mode="r")
    dcnn_images = np.load(dcnn_images_path, mmap_mode="r")

    test_indices = np.load(test_indices_path)

    if len(labels) != len(features):
        raise ValueError("AFE feature and label counts differ.")

    if len(labels) != len(dcnn_images):
        raise ValueError("DCNN image and label counts differ.")

    if (
        len(test_indices) == 0
        or np.any(test_indices < 0)
        or np.any(test_indices >= len(labels))
    ):
        raise ValueError("Invalid test indices.")

    classes = np.unique(labels)

    # Load the authoritative category-to-label mapping
    # created during preprocessing.
    class_mapping_path = processed_dir / "class_mapping.json"

    if not class_mapping_path.exists():
        raise FileNotFoundError(
            f"Class mapping file not found: {class_mapping_path}"
        )

    with open(class_mapping_path, "r", encoding="utf-8") as file:
        class_mapping = json.load(file)

    # Convert category -> label into label -> category.
    label_to_category = {
        int(label): category
        for category, label in class_mapping.items()
    }

    # Ensure every numeric label has a corresponding category.
    if set(classes.tolist()) != set(label_to_category.keys()):
        raise ValueError(
            "Saved class mapping does not match the labels in the dataset."
        )

    class_names = [
        label_to_category[int(label)]
        for label in classes
    ]

    if len(classes) != len(class_names):
        raise ValueError(
            "Category names and numeric class labels do not match."
        )

    y_true = np.asarray(
        labels[test_indices],
    )

    if not np.isfinite(y_true).all():
        raise ValueError("Test labels contain invalid values.")

    evaluation_dir = models_dir / "evaluations"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Evaluate the three traditional ML models.
    X_test = np.asarray(
        features[test_indices],
        dtype=np.float32,
    )

    if not np.isfinite(X_test).all():
        raise ValueError("Test features contain NaN or infinity.")

    for model_key, model_name in CLASSICAL_MODELS.items():

        model_path = models_dir / f"{model_key}.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {model_path}"
            )

        model = joblib.load(model_path)

        predictions = model.predict(X_test)

        metrics = _evaluate_predictions(
            y_true,
            predictions,
            classes,
            class_names,
        )

        result = {
            "model_key": model_key,
            "model_name": model_name,
            "test_samples": len(test_indices),
            **metrics,
        }

        _save_json(
            result,
            evaluation_dir / f"{model_key}_evaluation.json",
        )

        np.save(
            evaluation_dir / f"{model_key}_confusion_matrix.npy",
            np.asarray(metrics["confusion_matrix"]),
        )

        results[model_key] = result

    # Evaluate DCNN.
    dcnn_model_path = dcnn_dir / "coin_dcnn.keras"

    if not dcnn_model_path.exists():
        raise FileNotFoundError(
            f"DCNN model not found: {dcnn_model_path}"
        )

    dcnn_model = tf.keras.models.load_model(
        dcnn_model_path
    )

    dcnn_predictions = _predict_dcnn(
        dcnn_model,
        dcnn_images,
        test_indices,
    )

    # DCNN outputs encoded class indices. Convert them back
    # to the original numeric labels used by the dataset.
    if (
        dcnn_predictions.min() < 0
        or dcnn_predictions.max() >= len(classes)
    ):
        raise ValueError("DCNN produced an invalid class index.")

    dcnn_predictions = classes[dcnn_predictions]

    dcnn_metrics = _evaluate_predictions(
        y_true,
        dcnn_predictions,
        classes,
        class_names,
    )

    dcnn_result = {
        "model_key": DCNN_MODEL_KEY,
        "model_name": DCNN_MODEL_NAME,
        "test_samples": len(test_indices),
        **dcnn_metrics,
    }

    _save_json(
        dcnn_result,
        evaluation_dir / "dcnn_evaluation.json",
    )

    np.save(
        evaluation_dir / "dcnn_confusion_matrix.npy",
        np.asarray(dcnn_metrics["confusion_matrix"]),
    )

    results[DCNN_MODEL_KEY] = dcnn_result

    # Compare all four models using macro F1-score.
    comparison = []

    for model_key, result in results.items():
        comparison.append({
            "model_key": model_key,
            "model_name": result["model_name"],
            "accuracy": result["accuracy"],
            "precision_macro": result["precision_macro"],
            "recall_macro": result["recall_macro"],
            "f1_macro": result["f1_macro"],
        })

    best_model = max(
        comparison,
        key=lambda item: item["f1_macro"],
    )

    comparison_result = {
        "dataset_id": dataset_id,
        "test_samples": len(test_indices),
        "selection_metric": "macro_f1",
        "best_model": best_model,
        "models": comparison,
    }

    _save_json(
        comparison_result,
        evaluation_dir / "comparison.json",
    )

    comparison_plot = _save_comparison_plot(
        comparison_result,
        evaluation_dir,
    )

    return {
        "individual_results": results,
        "comparison": comparison_result,
        "comparison_plot": comparison_plot,
        "evaluation_directory": str(evaluation_dir),
    }



def load_evaluation_results(dataset_id):
    """Load previously saved evaluation results."""

    evaluation_dir = (
        Path(MODELS_DIR)
        / dataset_id
        / "evaluations"
    )

    comparison_path = evaluation_dir / "comparison.json"

    if not comparison_path.exists():
        return None

    with open(comparison_path, "r", encoding="utf-8") as file:
        comparison = json.load(file)

    individual_results = {}

    for model_key in [
        "naive_bayes",
        "decision_tree",
        "random_forest",
        "dcnn",
    ]:
        result_path = (
            evaluation_dir
            / f"{model_key}_evaluation.json"
        )

        if result_path.exists():
            with open(result_path, "r", encoding="utf-8") as file:
                individual_results[model_key] = json.load(file)

    return {
        "comparison": comparison,
        "individual_results": individual_results,
        "evaluation_directory": str(evaluation_dir),
    }
