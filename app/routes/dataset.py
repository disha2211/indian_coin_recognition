from uuid import uuid4
from app.services.afe_service import extract_afe_dataset
from app.services.split_service import create_or_load_split

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

from app.config import UPLOAD_DIR
from app.services.dataset_service import inspect_dataset
from app.services.preprocessing_service import preprocess_dataset
from app.services.state_service import load_project_state, save_project_state

bp = Blueprint("dataset", __name__)


def allowed_dataset(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "zip"

@bp.get("/")
def index():
    dataset = current_app.config.get("CURRENT_DATASET")
    preprocessing = current_app.config.get("CURRENT_PREPROCESSING")
    afe = current_app.config.get("CURRENT_AFE")
    split = current_app.config.get("CURRENT_SPLIT")

    if dataset is None:
        state = load_project_state()

        if state:
            dataset = state.get("dataset")
            preprocessing = state.get("preprocessing")
            afe = state.get("afe")

            if dataset:
                current_app.config["CURRENT_DATASET"] = dataset

            if preprocessing:
                current_app.config["CURRENT_PREPROCESSING"] = preprocessing

            if afe:
                current_app.config["CURRENT_AFE"] = afe

    return render_template(
        "dataset.html",
        dataset=dataset,
        preprocessing=preprocessing,
        afe=afe,
        split=split,
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
