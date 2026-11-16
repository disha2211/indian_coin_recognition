from pathlib import Path

import joblib
import tensorflow as tf

from app.config import MODELS_DIR, PROCESSED_DIR


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


def inspect_saved_models(dataset_id):
    """
    Verify that all trained model artifacts exist and
    can be loaded successfully.
    """

    models_dir = Path(MODELS_DIR) / dataset_id
    dcnn_dir = Path(PROCESSED_DIR) / dataset_id / "dcnn"

    results = {}

    # -------------------------------------------------
    # Classical ML models
    # -------------------------------------------------

    for model_key, model_info in CLASSICAL_MODELS.items():

        model_path = models_dir / model_info["filename"]

        result = {
            "model_key": model_key,
            "model_name": model_info["name"],
            "filename": model_info["filename"],
            "path": str(model_path),
            "exists": model_path.exists(),
            "loadable": False,
            "status": "missing",
        }

        if model_path.exists():
            try:
                joblib.load(model_path)

                result["loadable"] = True
                result["status"] = "ready"

            except Exception as exc:
                result["status"] = "load_error"
                result["error"] = str(exc)

        results[model_key] = result

    # -------------------------------------------------
    # DCNN
    # -------------------------------------------------

    dcnn_path = dcnn_dir / "coin_dcnn.keras"

    dcnn_result = {
        "model_key": "dcnn",
        "model_name": "AFE-Enhanced Deep Convolutional Neural Network",
        "filename": "coin_dcnn.keras",
        "path": str(dcnn_path),
        "exists": dcnn_path.exists(),
        "loadable": False,
        "status": "missing",
    }

    if dcnn_path.exists():
        try:
            tf.keras.models.load_model(dcnn_path)

            dcnn_result["loadable"] = True
            dcnn_result["status"] = "ready"

        except Exception as exc:
            dcnn_result["status"] = "load_error"
            dcnn_result["error"] = str(exc)

    results["dcnn"] = dcnn_result

    return results