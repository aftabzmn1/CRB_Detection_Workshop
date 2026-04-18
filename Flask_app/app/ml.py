"""
Handles YOLO model loading and detection logic.

"""

from typing import Any, Optional
# Used for type hints (helps readability, not required for execution)

import cv2
# OpenCV used to save images after drawing bounding boxes

from app.config import (
    ALLOWED_EXTENSIONS,   # allowed image formats
    DETECTED_DIR,         # where detected images will be saved
    TRAP_IMAGES_DIR,      # where input images are stored
    WEIGHTS_PATH,         # path to YOLO model file
)

# Global variable to store loaded YOLO model (so it loads only once)
_model = None

# Stores error message if model fails to load
_load_error: Optional[str] = None


def _inference_device() -> str:
    """CUDA if available, else Apple MPS, else CPU (Ultralytics device string)."""
    import torch

    if torch.cuda.is_available():
        return "0"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def load_model() -> None:
    """Load YOLO model only once into memory."""
    global _model, _load_error

    # If already loaded or failed before, do nothing
    if _model is not None or _load_error is not None:
        return

    # Check if model file exists
    if not WEIGHTS_PATH.is_file():
        _load_error = f"No weights file at {WEIGHTS_PATH}. Add best.pt to weights/"
        return

    try:
        from ultralytics import YOLO
        # Import YOLO model from Ultralytics

        _model = YOLO(str(WEIGHTS_PATH))
        # Load model into memory

    except Exception as e:
        _load_error = f"Failed to load model: {e}"
        # Store error if loading fails


def get_load_error() -> Optional[str]:
    """Return error message if model failed to load."""
    return _load_error


def list_trap_image_paths():
    """Get all valid image files from input folder."""
    
    # Ensure folder exists
    TRAP_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    paths = []

    # Loop through all files in folder
    for p in sorted(TRAP_IMAGES_DIR.iterdir()):
        # Check file and allowed extension
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS:
            paths.append(p)

    return paths


def build_detection_results() -> tuple[Optional[str], list[dict[str, Any]]]:
    """
    Run YOLO on all images and return results for frontend.
    """

    # Ensure model is loaded
    load_model()

    # If loading failed, return error
    if _load_error:
        return _load_error, []

    # Model must exist at this point
    assert _model is not None

    # Ensure output folder exists
    DETECTED_DIR.mkdir(parents=True, exist_ok=True)

    # Get input images
    paths = list_trap_image_paths()

    results: list[dict[str, Any]] = []

    # Get class names from model (e.g., crb, leaf)
    names = getattr(_model, "names", None) or {}

    # Process each image
    for path in paths:
        try:
            # Run YOLO prediction (device auto: NVIDIA / Apple GPU / CPU)
            pred = _model.predict(
                source=str(path), verbose=False, device=_inference_device()
            )

            # Get result for one image
            r = pred[0]

            # Draw bounding boxes on image
            annotated = r.plot()

            # Create output filename
            out_name = f"{path.stem}_detected.jpg"
            out_path = DETECTED_DIR / out_name

            # Save detected image
            cv2.imwrite(str(out_path), annotated)

            labels: list[str] = []

            # Extract detection labels
            if r.boxes is not None and len(r.boxes):
                for box in r.boxes:
                    cls_id = int(box.cls[0])   # class id
                    conf = float(box.conf[0]) # confidence
                    label = names.get(cls_id, str(cls_id))
                    labels.append(f"{label}: {conf:.2f}")

            # If no detection found
            if not labels:
                labels = ["No detections."]

            # Store result for frontend
            results.append(
                {
                    "source_name": path.name,
                    "url": f"detected/{out_name}",
                    "labels": labels,
                    "error": None,
                }
            )

        except Exception as e:
            # Handle errors per image
            results.append(
                {
                    "source_name": path.name,
                    "url": None,
                    "labels": [],
                    "error": str(e),
                }
            )

    return None, results