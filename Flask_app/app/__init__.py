"""
Flask application factory — builds and configures the Flask app.
run.py calls create_app() to start the server.
"""

from pathlib import Path
# Used to handle file system paths in a clean and cross-platform way

from flask import Flask
# Import Flask class to create the web application


def create_app() -> Flask:
    # Get project root directory (one level above app/)
    # This is where folders like "static/" and "weights/" exist
    root = Path(__file__).resolve().parent.parent

    # Create Flask app instance
    app = Flask(
        __name__,  # name of current module (used internally by Flask)

        # Location of HTML template files
        template_folder=str(Path(__file__).parent / "templates"),

        # Location of static files (images, CSS, uploads, detected outputs)
        static_folder=str(root / "static"),

        # URL path to access static files in browser (/static/...)
        static_url_path="/static",
    )

    # Import blueprint (collection of routes) from routes.py
    from app.routes import bp

    # Register blueprint to connect routes with this app
    app.register_blueprint(bp)

    # Return fully configured Flask app
    return app