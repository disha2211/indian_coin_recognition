from flask import Flask

from app.config import MAX_CONTENT_LENGTH


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY="change-this-in-production",
        MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH,
    )

    from app.routes.dataset import bp as dataset_bp
    app.register_blueprint(dataset_bp, url_prefix="/dataset")

    @app.get("/")
    def dashboard():
        return (
            "<h1>Indian Coin Recognition System</h1>"
            "<p><a href='/dataset/'>Dataset Upload & Preview</a></p>"
        )

    @app.errorhandler(413)
    def too_large(_error):
        return "Dataset is too large. Maximum upload size is 512 MB.", 413

    return app
