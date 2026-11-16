from flask import Flask, render_template

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
        app.config["CURRENT_AFE"] = state.get("afe")
        app.config["CURRENT_SPLIT"] = state.get("split")

    from app.routes.dataset import bp as dataset_bp
    app.register_blueprint(dataset_bp, url_prefix="/dataset")

    @app.get("/")
    def dashboard():
        state = load_project_state() or {}

        dataset = state.get("dataset")
        preprocessing = state.get("preprocessing")
        afe = state.get("afe")
        split = state.get("split")
        models = state.get("models") or {}
        dcnn = state.get("dcnn")
        evaluation = state.get("evaluation")
        prediction = state.get("prediction")

        training_done = bool(models or dcnn)
        stages = [
            {"label": "1. Dataset upload", "anchor": "dataset-upload", "done": bool(dataset)},
            {"label": "2. Image preprocessing", "anchor": "preprocessing", "done": bool(preprocessing)},
            {"label": "3. Adaptive feature enhancement", "anchor": "afe", "done": bool(afe)},
            {"label": "4. Train/test split", "anchor": "split", "done": bool(split)},
            {"label": "5. Model training", "anchor": "model-training", "done": training_done},
            {"label": "6. Model evaluation", "anchor": "model-evaluation", "done": bool(evaluation)},
            {"label": "7. Coin prediction", "anchor": "prediction", "done": bool(prediction)},
        ]

        return render_template(
            "dashboard.html",
            dataset=dataset,
            evaluation=evaluation,
            saved_model_count=len(models) + (1 if dcnn else 0),
            stages=stages,
            completed_stages=sum(1 for item in stages if item["done"]),
        )

    @app.errorhandler(413)
    def too_large(_error):
        limit_gib = MAX_CONTENT_LENGTH / (1024 ** 3)
        return (
            f"Dataset is too large. Maximum upload size is "
            f"{limit_gib:.1f} GiB. Increase MAX_UPLOAD_BYTES if needed."
        ), 413

    return app