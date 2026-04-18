# AI-Powered CRB Detection (YOLOv8 + Flask)

## Workshop overview

This repository supports a workshop led by **Mohammad Aftab Uzzaman** as part of the **Hawaiʻi Data Science Institute (HIDSI)** Fellowship Program. The workshop shows how computer vision can help detect the **Coconut Rhinoceros Beetle (CRB)** in camera trap imagery.

---

## Motivation: Coconut Rhinoceros Beetle (CRB)

The Coconut Rhinoceros Beetle (CRB) is a major pest in Hawaiʻi. It causes serious damage to palm trees. If the population is not controlled, the damage can spread quickly. In the field, CRB damage can be seen as bore holes, frass, and V-shaped cuts on leaves. These signs are easy to recognize once you know what to look for.

However, checking for these signs in images is not easy. Many cameras are placed in the field and they capture images for long periods of time. This creates a large number of images. Going through each image manually takes a lot of time and effort. It is also easy to miss important details when the dataset becomes too large.

Because of this, manual monitoring does not scale well. Automated systems can help solve this problem. By combining field or IoT cameras with detection models, we can automatically identify images that may contain CRB. This helps reduce manual work and allows faster and more efficient monitoring.

![Figure 1 — Pheromone trap used for CRB monitoring](docs/Images/pheromon_trap.png)

*Courtesy: Dr. Mohsen Paryavi.*

![Figure 2 — Palm tree damage associated with CRB activity](docs/Images/tree_damage.png)

*Courtesy: Dr. Mohsen Paryavi.*

---

## Object detection and YOLO

Object detection means we identify what is in an image and also where it is. The model draws boxes around each object. Each box has a position and a size. The position is defined by x and y, which represent the center of the box. The size is defined by width (w) and height (h). The model also gives a confidence score to show how sure it is about each detection.

This is important for this project because a single image can contain multiple objects such as CRB, leaves, and background. The model must detect each object separately and show its exact location.

YOLO stands for You Only Look Once. This means the model looks at the image one time and makes predictions. Because of this, YOLO is very fast and efficient. It is suitable for real-time applications and large image datasets.

In this workshop, we use Ultralytics YOLOv8. YOLOv8 is already trained on a large dataset called COCO, which contains many common objects. We fine-tune this model using our CRB dataset so it can detect specific classes like CRB, leaf, and other.

YOLO models come in different sizes such as nano (n), small (s), medium (m), large (l), and extra-large (x). The nano model (YOLOv8n) is very fast but less accurate. It is useful when speed is more important than accuracy. The medium model (YOLOv8m) provides a better balance between speed and accuracy. In this project, we use YOLOv8m because it gives good accuracy while still being fast enough for practical use.

How YOLO reads annotations:

Each image has a corresponding text file with the same name. For example, image1.jpg will have image1.txt. This file contains the annotation information. Each line in the file represents one object in the image.

The format is:

```
class x y w h
```

The class represents the object type, such as CRB, leaf, or other. The values x and y represent the center of the bounding box. The values w and h represent the width and height of the box. This is how YOLO understands what to detect and where to detect it.

![Figure 3 — Object detection with bounding boxes on trap imagery](docs/Images/object_detection.png)

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

After training, export **`best.pt`** from the run you want to use in the Flask app.

---

## Model results and evaluation

The plots below come from an Ultralytics training run using the validation dataset. These plots help us understand how the model performs overall and for each class.

The confusion matrix shows how well the model is making predictions. The diagonal values represent correct predictions. Higher values on the diagonal mean better performance.

From the results, the model detects CRB well. It correctly identifies most CRB samples. However, the performance for leaf is weaker. Many leaf samples are missed and classified as background. This means the model has lower recall for leaf.

There are also some false positives. In some cases, the model incorrectly predicts CRB or leaf when the image is actually background. This shows that the model can confuse objects with similar patterns.

![Figure 4 — Confusion matrix (normalized) for validation predictions](docs/Images/confusion_matrix.png)

Bias and limitations. The sample used here is only a small part of a larger monitoring effort. In this subset, there are more CRB examples than leaf. The class balance also varies across train, validation, and test sets.

Models usually perform better on classes with more data. This is why CRB performance is stronger. Leaf performance is weaker because there are fewer training examples.

Because of this, the results should be interpreted carefully. These results are useful for understanding the model behavior in this workshop. However, they do not fully represent real-world conditions across all locations in Hawaiʻi.

![Figure 5 — Overall model performance (aggregate detection metrics)](docs/Images/overall_model_performance.png)

Overall model performance shows the main metrics such as precision and recall. These values are calculated across different IoU thresholds using the validation dataset. It gives a general idea of how well the model is performing before looking at each class separately.

![Figure 6 — Class-wise performance (precision, recall, mAP per class)](docs/Images/Classwise_performance.png)

Class-wise performance shows the results for each class, such as CRB, leaf, and other. This helps us understand how the model behaves for different objects. In this dataset, CRB has more samples than leaf. Because of this, the model performs better for CRB and weaker for leaf. Classes with fewer examples can show lower performance and more variation in results.

---

## Local prediction with Flask

The trained weights **`best.pt`** are loaded by a small **Flask** app that lists trap images from **`Flask_app/static/uploads/images/`**, runs **YOLO** inference when you open the detections page, saves annotated images to **`Flask_app/static/detected/`**, and shows results in the browser. Weights are loaded from [`Flask_app/weights/best.pt`](Flask_app/weights/best.pt) (see [`Flask_app/app/config.py`](Flask_app/app/config.py)).

![Figure 7 — Flask app running on the local machine (browser)](docs/Images/webpage_running_on_local_machin.png)

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
├── docs/
│   ├── Images/              # README figures
│   └── HIDSI Workshop.pptx
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

## Acknowledgments

I thank Dr. Daniel Jenkins and Dr. Mohsen Paryavi for their guidance and support connected to this work. I am grateful to the Hawaiʻi Data Science Institute (HIDSI) and the Hawaiʻi State Energy Office (HSEO) for funding.
