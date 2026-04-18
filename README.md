# AI-Powered CRB Detection (YOLOv8 + Flask)

## Workshop overview

This repository supports a workshop led by **Mohammad Aftab Uzzaman** as part of the **Hawaiʻi Data Science Institute (HIDSI)** Fellowship Program. The workshop shows how computer vision can help detect the **Coconut Rhinoceros Beetle (CRB)** in camera trap imagery.

---

## Motivation: Coconut Rhinoceros Beetle (CRB)

- CRB is a major pest in Hawaiʻi.
- It can severely damage palm trees.
- Field signs include bore holes and characteristic feeding damage.
- Manual review of trap photos is slow and labor-intensive.
- Automated pipelines (e.g., IoT cameras + detection models) can scale monitoring.

> **Figure (optional):** Add photos of CRB damage on palms for documentation.  
> *Suggested credit: Dr. Mohsen Paryavi (per workshop materials).*

---

## Object detection and YOLO (short)

- **Object detection** combines *what* is in the image (classification) with *where* it is (bounding boxes).
- Models output **bounding boxes**, **class labels**, and **confidence scores**.
- **YOLO** (“You Only Look Once”) performs fast, single-pass detection; this project uses **Ultralytics YOLOv8**.

> **Figure (optional):** Add an example image with bounding boxes drawn around CRB / other classes.

---

## Model training (Google Colab)

Training is done in **Google Colab** with GPU. The notebook in this repo is:

| File | Role |
|------|------|
| [`Training_YOLOV8_Colab.ipynb`](Training_YOLOV8_Colab.ipynb) | End-to-end Colab workflow: Drive paths, dataset prep, YOLOv8 training |

The notebook is written for a **Colab + Google Drive** layout (see paths inside the notebook). Adjust paths if your Drive folder names differ.

### Dataset layout (YOLO)

The workshop dataset under [`Dataset/`](Dataset/) follows the usual YOLO layout:

- [`Dataset/images/`](Dataset/images/) — images (e.g. `.jpg`)
- [`Dataset/labels/`](Dataset/labels/) — one `.txt` label file per image (same base name)
- [`Dataset/crb.yaml`](Dataset/crb.yaml) — data YAML with **train / val / test** image paths and class names:

  - `0`: **crb**
  - `1`: **leaf**
  - `2`: **other**

In Colab, the notebook copies data from Drive into a local folder (e.g. `/content/crb_dataset_local/`), builds **train / val / test** splits (**70% / 20% / 10%**, `random.seed(42)`), and trains with `crb.yaml`.

### Training steps (from the notebook)

1. Install **ultralytics** and set **source** paths (Drive) and **local** working directory.
2. Create `train`, `val`, and `test` folders (each with `images/` and `labels/`).
3. Shuffle and split image filenames; copy each image and matching `.txt` label into the split folders; copy `crb.yaml` locally.
4. Load a pretrained **`yolov8m.pt`** model and call `model.train(...)` with parameters such as `epochs`, `imgsz=640`, `batch`, and `device` appropriate for Colab GPU (`cuda` in the notebook).

### Evaluation

The notebook relies on **Ultralytics** training **validation** and saved **plots** under the run directory (as configured by `project` / `name` in `model.train`). After training, export **`best.pt`** from the run you want to use in the Flask app.

> **Note:** This README does not embed numeric metrics or confusion matrices. For numbers and figures, use your Colab run outputs or the workshop slides in [`docs/`](docs/).

---

## Model results

Training artifacts (curves, confusion matrix, batch images, etc.) are produced **inside Colab** by Ultralytics when you run the training cells—check the run folder printed during training (e.g. under `runs/detect/...` or the `project` path you set).

If something is missing in the notebook outputs, use the slides in **`docs/`** for the workshop’s explanation of results and concepts.

---

## Local prediction with Flask

The trained weights **`best.pt`** are loaded by a small **Flask** app that:

- Lists trap images from **`Flask_app/static/uploads/images/`**
- Runs **YOLO** inference when you open the detections page
- Saves annotated images to **`Flask_app/static/detected/`** and shows them in the browser

Weights path used by the app: [`Flask_app/weights/best.pt`](Flask_app/weights/best.pt) (see [`Flask_app/app/config.py`](Flask_app/app/config.py)).

> **Figure (optional):** Screenshot of the app at `http://127.0.0.1:5000`.

---

## How to run the Flask app

All commands below assume you start from the **`Flask_app/`** directory (where `run.py` and `requirements.txt` live).

### macOS / Linux

```bash
cd Flask_app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

### Windows (Command Prompt / PowerShell)

```bash
cd Flask_app
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

Then open **http://127.0.0.1:5000** in your browser.

**What this does:** create an isolated Python environment, install **Flask**, **ultralytics**, **torch**, **OpenCV**, etc., start the dev server on port **5000**, and load the model once at startup (`run.py` calls `load_model()` before `app.run(...)`).

---

## Flask execution flow (how the code fits together)

1. **`Flask_app/run.py`** — Builds the app with `create_app()`, loads the YOLO model once via `load_model()`, runs Flask on `127.0.0.1:5000` with `debug=True`.
2. **`Flask_app/app/__init__.py`** — App factory: sets templates and `static/`, registers the blueprint from `routes.py`.
3. **`Flask_app/app/routes.py`** — Routes:
   - **`/`** — Lists images under `static/uploads/images/` (no inference).
   - **`/detected`** — Calls `build_detection_results()` in `ml.py` to run inference and render results.
4. **`Flask_app/app/ml.py`** — Loads **`weights/best.pt`** with Ultralytics **YOLO**, picks **CUDA → Apple MPS → CPU** when predicting, draws boxes with `r.plot()`, saves `*_detected.jpg` under `static/detected/`, and passes URLs and label strings to the template.
5. **`Flask_app/app/config.py`** — Central paths: `WEIGHTS_PATH`, `TRAP_IMAGES_DIR`, `DETECTED_DIR`, allowed image extensions.

---

## Repository layout (summary)

```
CRB_Detection_Workshop/
├── README.md
├── Training_YOLOV8_Colab.ipynb
├── Dataset/
│   ├── images/
│   ├── labels/
│   └── crb.yaml
├── docs/                    # workshop slides (e.g. HIDSI Workshop.pptx)
└── Flask_app/
    ├── run.py
    ├── requirements.txt
    ├── weights/best.pt
    ├── app/
    │   ├── __init__.py
    │   ├── routes.py
    │   ├── ml.py
    │   ├── config.py
    │   └── templates/
    └── static/
        ├── uploads/images/
        └── detected/
```

---

## Workshop materials

- **Slides:** [`docs/`](docs/) (PowerPoint). Use them for narrative, results context, and concepts if the notebook outputs are incomplete.

---

## Acknowledgments

- Dr. Daniel Jenkins  
- Dr. Mohsen Paryavi  
- Hawaiʻi Data Science Institute (HIDSI)  
- Hawaiʻi State Energy Office  

---

## Notes for reviewers

- Training is intended to run in **Google Colab** (GPU); paths in the notebook target **Drive + local Colab** folders.
- The **Flask** app runs **locally** and reads **`Flask_app/weights/best.pt`**.
- Dataset labels follow **YOLO** format (`Dataset/images` + matching `Dataset/labels` `.txt` files, `Dataset/crb.yaml` for class names and split paths in Colab).
