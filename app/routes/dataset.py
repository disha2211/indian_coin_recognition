from uuid import uuid4
from app.services.afe_service import extract_afe_dataset
from app.services.split_service import create_or_load_split
from app.services.model_training_service import train_classical_model

from app.services.evaluation_service import (
    evaluate_all_models,
    load_evaluation_results,
)
from flask import send_file

from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.utils import secure_filename
import json
from pathlib import Path

from app.config import PROCESSED_DIR

from app.config import UPLOAD_DIR
from app.services.dataset_service import inspect_dataset
from app.services.preprocessing_service import preprocess_dataset
from app.services.state_service import load_project_state, save_project_state

bp = Blueprint("dataset", __name__)


def allowed_dataset(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "zip"

@bp.get("/")
def index():
    # The saved project state is the source of truth.
    state = load_project_state() or {}

    dataset = state.get("dataset")
    preprocessing = state.get("preprocessing")
    afe = state.get("afe")
    split = state.get("split")
    models = state.get("models", {})
    dcnn = current_app.config.get("CURRENT_DCNN")
    evaluation = state.get("evaluation")

    # Restore all modules into Flask application state.
    current_app.config["CURRENT_DATASET"] = dataset
    current_app.config["CURRENT_PREPROCESSING"] = preprocessing
    current_app.config["CURRENT_AFE"] = afe
    current_app.config["CURRENT_SPLIT"] = split
    current_app.config["CURRENT_MODELS"] = models
    current_app.config["CURRENT_EVALUATION"] = evaluation

    # Load saved DCNN metadata if it is not already in memory.
    if dataset and dcnn is None:
        dcnn_metadata_path = (
        Path(PROCESSED_DIR)
        / dataset["dataset_id"]
        / "dcnn"
        / "dcnn_model_metadata.json"
    )

    if dcnn_metadata_path.exists():
        try:
            with open(
                dcnn_metadata_path,
                "r",
                encoding="utf-8",
            ) as file:
                dcnn = json.load(file)

            current_app.config["CURRENT_DCNN"] = dcnn

        except (OSError, json.JSONDecodeError):
            dcnn = None

    if dataset:
        saved_evaluation = load_evaluation_results(
        dataset["dataset_id"]
    )

    if saved_evaluation:
        evaluation = saved_evaluation
        current_app.config["CURRENT_EVALUATION"] = evaluation

    return render_template(
        "dataset.html",
        dataset=dataset,
        preprocessing=preprocessing,
        afe=afe,
        split=split,
        models=models,
        dcnn=dcnn,
        evaluation=evaluation,
    )

@bp.post("/upload")
def upload():
    uploaded = request.files.get("dataset")
    if not uploaded or not uploaded.filename:
        flash("Please select a ZIP dataset before uploading.", "error")
        return redirect(url_for("dataset.index"))

    if not allowed_dataset(uploaded.filename):
        flash("Invalid dataset format. Only ZIP files are supported.", "error")
        return redirect(url_for("dataset.index"))

    # Never trust the user-provided filename; use a unique server-side name.
    safe_name = secure_filename(uploaded.filename) or "dataset.zip"
    upload_path = UPLOAD_DIR / f"{uuid4().hex}_{safe_name}"

    try:
        # Werkzeug streams the incoming multipart file to its temporary storage;
        # save() copies it to disk rather than reading the entire archive into RAM.
        uploaded.save(upload_path)
        metadata = inspect_dataset(upload_path)

        current_app.config["CURRENT_DATASET"] = metadata
        current_app.config["CURRENT_PREPROCESSING"] = None

        save_project_state(
            {
                "dataset": metadata,
                "preprocessing": None,
            }
        )

        flash("Dataset uploaded, extracted and validated successfully.", "success")

        return render_template(
            "dataset.html",
            dataset=metadata,
            preprocessing=None,
        )
    except ValueError as exc:
        flash(str(exc), "error")
    except OSError:
        current_app.logger.exception("Dataset storage operation failed")
        flash("Not enough disk space or permission to store the dataset.", "error")
    except Exception:
        current_app.logger.exception("Dataset processing failed")
        flash("The dataset could not be processed. Check the ZIP structure and available disk space.", "error")
    finally:
        # Keep the ZIP only if required for audit/retry? For this phase, extracted
        # files and metadata are sufficient, so remove the temporary archive.
        if upload_path.exists():
            upload_path.unlink(missing_ok=True)

    return redirect(url_for("dataset.index"))

@bp.post("/preprocess")
def preprocess():
    dataset = current_app.config.get("CURRENT_DATASET")

    if not dataset:
        flash(
            "Please upload and validate a dataset before preprocessing.",
            "error",
        )
        return redirect(url_for("dataset.index"))

    try:
        metadata = preprocess_dataset(dataset)

        current_app.config["CURRENT_PREPROCESSING"] = metadata

        save_project_state(
            {
                "dataset": dataset,
                "preprocessing": metadata,
            }
        )

        flash("Image preprocessing completed successfully.", "success")

        return render_template(
            "dataset.html",
            dataset=dataset,
            preprocessing=metadata,
        )

    except ValueError as exc:
        current_app.logger.exception("Dataset preprocessing validation failed")
        flash(str(exc), "error")

    except Exception:
        current_app.logger.exception("Dataset preprocessing failed")
        flash(
            "Image preprocessing failed. Check the dataset and available disk space.",
            "error",
        )

    return redirect(url_for("dataset.index"))


@bp.post("/afe")
def run_afe():
    dataset = current_app.config.get("CURRENT_DATASET")
    preprocessing = current_app.config.get("CURRENT_PREPROCESSING")

    if not dataset:
        flash(
            "Please upload and validate a dataset before running AFE.",
            "error",
        )
        return redirect(url_for("dataset.index"))

    if not preprocessing:
        flash(
            "Please complete image preprocessing before running AFE.",
            "error",
        )
        return redirect(url_for("dataset.index"))

    try:
        metadata = extract_afe_dataset(dataset)

        current_app.config["CURRENT_AFE"] = metadata

        # Preserve the existing persistent state.
        state = load_project_state() or {}

        state["dataset"] = dataset
        state["preprocessing"] = preprocessing
        state["afe"] = metadata

        save_project_state(state)

        flash(
            "Adaptive Feature Enhancement completed successfully.",
            "success",
        )

        return render_template(
            "dataset.html",
            dataset=dataset,
            preprocessing=preprocessing,
            afe=metadata,
        )

    except FileNotFoundError as exc:
        flash(str(exc), "error")

    except ValueError as exc:
        current_app.logger.exception(
            "AFE validation failed"
        )
        flash(str(exc), "error")

    except Exception:
        current_app.logger.exception(
            "AFE processing failed"
        )
        flash(
            "Adaptive Feature Enhancement failed. "
            "Check the dataset and available disk space.",
            "error",
        )

    return redirect(url_for("dataset.index"))

@bp.post("/split")
def run_split():
    dataset = current_app.config.get("CURRENT_DATASET")
    preprocessing = current_app.config.get("CURRENT_PREPROCESSING")
    afe = current_app.config.get("CURRENT_AFE")

    if not dataset:
        flash("Please upload a dataset first.", "error")
        return redirect(url_for("dataset.index"))

    if not preprocessing or not afe:
        flash(
            "Complete preprocessing and AFE before creating the split.",
            "error",
        )
        return redirect(url_for("dataset.index"))

    try:
        metadata = create_or_load_split(dataset)

        current_app.config["CURRENT_SPLIT"] = metadata

        state = load_project_state() or {}
        state["dataset"] = dataset
        state["preprocessing"] = preprocessing
        state["afe"] = afe
        state["split"] = metadata

        save_project_state(state)

        if metadata.get("reused_existing"):
            flash("Existing valid train/test split loaded successfully.", "success")
        else:
            flash("Stratified train/test split created successfully.", "success")

        return render_template(
            "dataset.html",
            dataset=dataset,
            preprocessing=preprocessing,
            afe=afe,
            split=metadata,
        )

    except (FileNotFoundError, ValueError) as exc:
        current_app.logger.exception("Train/test split validation failed")
        flash(str(exc), "error")

    except Exception:
        current_app.logger.exception("Train/test split failed")
        flash(
            "Could not create the train/test split. Check the logs.",
            "error",
        )

    return redirect(url_for("dataset.index"))


@bp.post("/train/<model_key>")
def train_model(model_key):
    allowed_models = {
        "naive_bayes": "Multinomial Naive Bayes",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
    }

    if model_key not in allowed_models:
        flash("Invalid model selected.", "error")
        return redirect(url_for("dataset.index"))

    dataset = current_app.config.get("CURRENT_DATASET")
    afe = current_app.config.get("CURRENT_AFE")
    split = current_app.config.get("CURRENT_SPLIT")

    if not dataset:
        flash("Please upload a dataset first.", "error")
        return redirect(url_for("dataset.index"))

    if not afe or not split:
        flash(
            "Complete AFE and train/test splitting before training models.",
            "error",
        )
        return redirect(url_for("dataset.index"))

    try:
        force_retrain = request.form.get("force_retrain") == "1"

        result = train_classical_model(
            dataset_id=dataset["dataset_id"],
            model_key=model_key,
            force_retrain=force_retrain,
        )
        # Preserve the state of all previously completed modules.
        state = load_project_state() or {}

        models = state.get("models", {})
        models[model_key] = result

        state["dataset"] = dataset
        state["afe"] = afe
        state["split"] = split
        state["models"] = models

        save_project_state(state)

        current_app.config["CURRENT_MODELS"] = models

        if result["status"] == "reused":
            flash(
                f"{result['model_name']} loaded from saved artifacts.",
                "success",
            )
        else:
            flash(
                f"{result['model_name']} trained and saved successfully.",
                "success",
            )

    except (FileNotFoundError, ValueError) as exc:
        current_app.logger.exception("Model training validation failed")
        flash(str(exc), "error")

    except Exception:
        current_app.logger.exception(
            "Model training failed: %s", model_key
        )
        flash(
            f"{allowed_models[model_key]} training failed. Check the application logs.",
            "error",
        )

    return redirect(url_for("dataset.index"))

@bp.route("/dcnn-plot/<dataset_id>/<plot_name>")
def dcnn_plot(dataset_id, plot_name):

    allowed_plots = {
        "accuracy_plot.png",
        "loss_plot.png",
    }

    if plot_name not in allowed_plots:
        return "Invalid plot.", 404

    plot_path = (
        Path(PROCESSED_DIR)
        / dataset_id
        / "dcnn"
        / plot_name
    )

    if not plot_path.exists():
        return "DCNN plot not found.", 404

    return send_file(plot_path)

@bp.post("/evaluate")
def run_evaluation():
    dataset = current_app.config.get("CURRENT_DATASET")

    if not dataset:
        flash("Please upload a dataset first.", "error")
        return redirect(url_for("dataset.index"))

    preprocessing = current_app.config.get("CURRENT_PREPROCESSING")
    afe = current_app.config.get("CURRENT_AFE")
    split = current_app.config.get("CURRENT_SPLIT")

    if not preprocessing or not afe or not split:
        flash(
            "Complete preprocessing, AFE and train/test split first.",
            "error",
        )
        return redirect(url_for("dataset.index"))

    try:
        dataset_id = dataset["dataset_id"]

        result = evaluate_all_models(
            dataset_id=dataset_id,
            dataset_metadata=dataset,
        )

        # Keep only the saved comparison and directory in project state.
        evaluation_state = {
            "comparison": result["comparison"],
            "evaluation_directory": result["evaluation_directory"],
        }

        current_app.config["CURRENT_EVALUATION"] = evaluation_state

        state = load_project_state() or {}

        state["dataset"] = dataset
        state["preprocessing"] = preprocessing
        state["afe"] = afe
        state["split"] = split
        state["evaluation"] = evaluation_state

        save_project_state(state)

        flash(
            "All four models evaluated successfully.",
            "success",
        )

        return redirect(url_for("dataset.index"))

    except (FileNotFoundError, ValueError) as exc:
        current_app.logger.exception("Model evaluation validation failed")
        flash(str(exc), "error")

    except Exception:
        current_app.logger.exception("Model evaluation failed")
        flash(
            "Model evaluation failed. Check the application logs.",
            "error",
        )

    return redirect(url_for("dataset.index"))
