"""
Defines URL routes (which URL shows which page).
Blueprint groups related routes together.
"""

from flask import Blueprint, render_template
# Blueprint → groups routes
# render_template → loads HTML pages

from app.ml import build_detection_results, list_trap_image_paths
# Import ML functions:
# - list_trap_image_paths → get input images
# - build_detection_results → run YOLO and get outputs

# Create blueprint named "main"
# This name is used in templates (e.g., url_for('main.trap_images'))
bp = Blueprint("main", __name__)


@bp.route("/")
# Route for homepage
def trap_images():
    """Show original trap images (no ML here)"""

    # Get list of image file paths
    paths = list_trap_image_paths()

    # Convert paths into format usable by HTML
    # URL must match static folder structure
    images = [
        {"name": p.name, "url": f"uploads/images/{p.name}"}
        for p in paths
    ]

    # Render HTML page and pass images
    return render_template("trap_images.html", images=images)


@bp.route("/detected")
# Route for detected images page
def detected_images():
    """Run YOLO when this page is opened"""

    # Run detection (ML happens here)
    err, results = build_detection_results()

    # Render HTML page with results
    return render_template("detected.html", error=err, results=results)