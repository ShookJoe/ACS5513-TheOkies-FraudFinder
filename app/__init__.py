from flask import Flask


def create_app() -> Flask:
    app = Flask(__name__)

    from app.routes import bp as pages_bp
    from app.api import bp as api_bp

    app.register_blueprint(pages_bp)
    app.register_blueprint(api_bp, url_prefix="/api")
    return app
