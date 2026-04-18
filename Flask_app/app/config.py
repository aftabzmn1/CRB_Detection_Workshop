"""
Paths and constants for the whole project.

This file lives inside the app/ package. ROOT is the CRB_Flask folder (next to
weights/ and static/), not the app/ folder
"""

from pathlib import Path
# Used to handle file paths properly across OS (Mac, Windows, Linux)

# app/config.py -> parent is app/, parent.parent is project root (CRB_Flask/)
ROOT: Path = Path(__file__).resolve().parent.parent
# ROOT = main project folder (CRB_Flask/)
# Used as base path for all other folders

# Trained YOLO weights (copy best.pt from Colab training here)
WEIGHTS_PATH = ROOT / "weights" / "best.pt"
# Full path to YOLO model file

# Original trap photos for the gallery and for inference (students add files here)
TRAP_IMAGES_DIR = ROOT / "static" / "uploads" / "images"
# Folder containing input images

# Annotated outputs written by the model (one *_detected.jpg per source image)
DETECTED_DIR = ROOT / "static" / "detected"
# Folder where detected images will be saved

# Filenames we accept from the trap folder (lowercase check in ml.py)
ALLOWED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
# Allowed image formats (prevents unsupported files)