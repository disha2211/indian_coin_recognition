from uuid import uuid4

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from app.config import UPLOAD_DIR
from app.services.dataset_service import inspect_dataset

bp = Blueprint("dataset", __name__)


def allowed_dataset(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "zip"


@bp.get("/")
def index():
    return render_template("dataset.html", dataset=None)


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
        flash("Dataset uploaded, extracted and validated successfully.", "success")
        return render_template("dataset.html", dataset=metadata)
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
