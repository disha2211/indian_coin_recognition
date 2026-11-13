from pathlib import Path

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from app.config import ALLOWED_DATASET_EXTENSIONS, UPLOAD_DIR
from app.services.dataset_service import inspect_dataset

bp = Blueprint("dataset", __name__)


def allowed_dataset(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_DATASET_EXTENSIONS


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

    upload_path = UPLOAD_DIR / uploaded.filename
    uploaded.save(upload_path)

    try:
        metadata = inspect_dataset(upload_path)
        current_app.config["CURRENT_DATASET"] = metadata
        flash("Dataset uploaded and validated successfully.", "success")
        return render_template("dataset.html", dataset=metadata)
    except ValueError as exc:
        flash(str(exc), "error")
    except Exception:
        current_app.logger.exception("Dataset processing failed")
        flash("The dataset could not be processed. Check the ZIP structure and image files.", "error")
    return redirect(url_for("dataset.index"))
