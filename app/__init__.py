from flask import Flask

from app.config import MAX_CONTENT_LENGTH
from app.services.state_service import load_project_state


def create_app() -> Flask:
    app = Flask(__name__)

    app.config.update(
        SECRET_KEY="change-this-in-production",
        MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH,
    )

    state = load_project_state()

    if state:
        app.config["CURRENT_DATASET"] = state.get("dataset")
        app.config["CURRENT_PREPROCESSING"] = state.get("preprocessing")

    from app.routes.dataset import bp as dataset_bp
    app.register_blueprint(dataset_bp, url_prefix="/dataset")

    @app.get("/")
    def dashboard():
        dataset = app.config.get("CURRENT_DATASET")
        preprocessing = app.config.get("CURRENT_PREPROCESSING")

        dataset_status = (
            "Dataset loaded"
            if dataset
            else "No dataset uploaded"
        )

        preprocessing_status = (
            "Preprocessing completed"
            if preprocessing
            else "Not completed"
        )

        return f"""
        <h1>Indian Coin Recognition System</h1>

        <h2>Project Status</h2>

        <p><strong>Dataset:</strong> {dataset_status}</p>
        <p><strong>Preprocessing:</strong> {preprocessing_status}</p>

        <p>
            <a href="/dataset/">Dataset Upload & Preview</a>
        </p>
        """

    @app.errorhandler(413)
    def too_large(_error):
        limit_gib = MAX_CONTENT_LENGTH / (1024 ** 3)
        return (
            f"Dataset is too large. Maximum upload size is "
            f"{limit_gib:.1f} GiB. Increase MAX_UPLOAD_BYTES if needed."
        ), 413

    return app