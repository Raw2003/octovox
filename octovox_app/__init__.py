"""OCTOVOX Flask application package.

Exposes an app-factory (:func:`create_app`) so the server can be launched
from a thin top-level ``run.py`` without ``sys.path`` hacks or a monolithic
``app.py``. Templates and static assets are resolved from inside the package;
runtime input/output data lives under ``New_OCTOVOX/data``.
"""
from flask import Flask

from . import config


def create_app():
    """Build and configure the Flask application."""
    config.ensure_dirs()

    app = Flask(
        __name__,
        static_folder=str(config.STATIC_DIR),
        template_folder=str(config.TEMPLATE_DIR),
    )
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

    # Register route blueprints (import here to avoid circulars and to keep
    # the heavy DSP imports out of module import time until app construction).
    from .routes.pages import pages_bp
    from .routes.api import api_bp
    app.register_blueprint(pages_bp)
    app.register_blueprint(api_bp)

    return app
