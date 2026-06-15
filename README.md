# 🎯 Darts Score Detector

This project automatically detects and scores darts in real-time from any camera feed, webcam, video, or still image. It outputs scores in standard notation like `S20`, `T17`, `D5`, `B25` (outer bull), `B50` (inner bull), or `Miss`.

### How it scores
Instead of using a complex neural network to guess the scores, this tool uses a simple two-step approach:
1. **YOLO models** locate the board, the bullseye rings, and each dart tip on the screen.
2. **Standard geometry** calculates the exact distance and angle of the dart tip relative to the bullseye, checking a lookup table to get the score.

This math-based approach makes the scoring highly reliable, transparent, and easy to debug.



---


## Table of contents

1. [How it works (architecture)](#how-it-works-architecture)
2. [Installation](#installation)
3. [Running it](#running-it)
4. [Annotating images in Roboflow](#annotating-images-in-roboflow)
5. [Re-training the models](#re-training-the-models)
6. [Technology stack](#technology-stack)



---


## How it works (architecture)

The pipeline is a **dual‑model hybrid** plus a deterministic geometric scorer:

```
                       ┌──────────────────────────────────────────┐
   frame ──────────────┤  YOLO11-seg  (models/yolo/best.pt)         │
                       │  5 classes: dartboard, 25-point circle,    │
                       │  50-point circle, twenty (double-20 arc),  │
                       │  arrow (dart)                              │
                       └──────────────┬───────────────────────────┘
                                      │ board ellipse, bull centre + radius,
                                      │ rotation landmark, dart polygons
                                      ▼
                       ┌──────────────────────────────────────────┐
   frame ──────────────┤  YOLO11-pose (models/yolo/pose_best.pt)    │
                       │  1 class "dart", 2 keypoints: tip + tail   │
                       └──────────────┬───────────────────────────┘
                                      │ precise dart tip pixel
                                      ▼
        ┌─────────────────────────────────────────────────────────┐
        │  Geometric scorer (score_geometric)                      │
        │   • homography rectification of the board plane          │
        │   • (distance ÷ bull-radius, angle) → ring + sector      │
        │   • regulation ring radii + 20-sector lookup table       │
        └─────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                         "T20", "D5", "B50", "Miss", …
```

**Why two models?**
- The **segmentation model** finds the board, the bull (the scoring origin and
  scale unit), the double‑20 arc (used to auto‑rotate the board so "20" sits at
  the top), and a coarse dart polygon.
- The **pose model** pinpoints the **dart tip** — a tiny feature that segmentation
  masks localise poorly. If no pose weights are present, the detector falls back
  to estimating the tip from the segmentation polygon (`dart_tip_tail_poly`).

**Geometric correction:** Since a camera is rarely perfectly aligned, the board looks like an oval (ellipse). The detector fixes this tilt, converting it back to a flat circle to ensure dart scoring (especially near the edge rings) is highly accurate. It also smooths out camera jitter across frames and votes on the final score to ensure video stability.



---


## Installation

```bash
# 1. clone
git clone https://github.com/raresgherasa/Darts-Score-Detector.git
cd "Darts Score"

# 2. create + activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt
```



---


## Running it

Entry point: **`darts_score_detection_offline.py`**.

```bash
# Webcam (device index 0)
python darts_score_detection_offline.py --input 0

# A video file
python darts_score_detection_offline.py --input path/to/game.mp4

# A single still image
python darts_score_detection_offline.py --input path/to/photo.jpg

# An IP camera / RTSP stream (Option 1: environment variable / .env)
# Create a gitignored .env file in the repository root:
# RTSP_URL="rtsp://user:pass@host:554/stream"
# Then run:
python darts_score_detection_offline.py

# An IP camera / RTSP stream (Option 2: direct command-line override)
python darts_score_detection_offline.py --input "rtsp://user:pass@host:554/stream"

# Write an annotated output video
python darts_score_detection_offline.py --input game.mp4 --output output_game.mp4
```

**Controls (live/video window):**
| Key | Action |
|-----|--------|
| `q` | quit |
| `SPACE` | (with `--capture`) save the current **raw** frame to `data/images/` for re‑annotation |



### Quick image smoke‑test

`test.py` renders the full detection + scoring overlay on still images (scoring
rings are drawn by default; pass `--no-rings` to hide them) — handy for visually
validating a freshly trained model:

```bash
python test.py --all                    # cycle through data/images/, any key = next, q = quit
python test.py --image path/to/img.jpg  # a single image
python test.py --all --save out_dir/    # headless: write annotated images instead of showing
python test.py --all --no-rings         # hide the scoring-ring overlay
```



---


## Annotating images in Roboflow

The segmentation model is trained on **instance‑segmentation polygons** exported from [Roboflow](https://roboflow.com).

![Roboflow Annotations Example](assets/roboflow_annotations.png)

### 1. Collect images
Photograph/clip your real board from the camera angle you'll deploy. Vary dart
positions, lighting, and number of darts. `--capture` (above) is the easy way to
collect on‑device frames into `data/images/`.

### 2. Create a Roboflow project
- Project type: **Instance Segmentation**.
- Upload your images.

### 3. Define exactly these 5 classes (names matter)
The code resolves classes **by name** (not index), accepting these labels:

| Class name (as exported) | What to draw | Role in scoring |
|--------------------------|--------------|-----------------|
| `dartboard` | polygon around the whole playable board face | board ellipse → perspective/homography + rotation |
| `The 25-point circle` | the outer‑bull ring | **scoring origin + unit radius** |
| `The 50-point circle` | the inner bullseye | truest board centre |
| `twenty` | the **double‑20 arc band** at the top (not the full wedge) | rotation landmark to put "20" at top |
| `arrow` | tight polygon tracing each dart, **from flight to tip** | dart, converted to a tip keypoint |

> Class names are matched flexibly (`resolve_class`), so the legacy 3‑class export
> `arrow` / `board` / `bulls` still works for inference. New datasets should use
> the 5‑class names above.

**Annotation tips**
- Trace the **arrow** polygon along the dart's true long axis (flight → barrel →
  tip). The flight (wide) vs tip (narrow) asymmetry is what lets the converter
  tell the tip from the tail.
- Keep polygons tight; avoid stray 1–2 px clicks (those become degenerate labels
  and are filtered out as annotation noise).
- The `twenty` class is the thin **double‑20 arc**, not the entire 20 wedge.

### 4. Generate + export
- Generate a dataset version. (Augmentation can be left to the training script,
  which already applies mosaic/mixup/copy‑paste/HSV/rotation/etc.)
- Export format: **YOLO11** (or "YOLOv8" segmentation — same `.txt` polygon
  format). Download the zip.

### 5. Unzip into the repo
Unzip so you get `train/images`, `train/labels`, and a `data.yaml`. Name the
folder `darts_dataset/` (the training default) — it is gitignored. If `valid/`
and `test/` splits are missing, the training script creates them automatically.



---


## Re-training the models

There are **two** models. Retrain the segmentation model whenever you change classes or add board imagery; regenerate + retrain the pose model whenever the segmentation labels change (the pose labels are derived from them).

### A. Segmentation model (board, bull, rings, dart)

```bash
python yolo_training.py --dataset darts_dataset --epochs 150
```

### B. Pose model (dart tip + tail)

**Step 1: Convert segmentation dataset to pose dataset**
```bash
python convert_seg_to_pose.py --src darts_dataset --dst darts_pose_dataset
```

**Step 2: Train pose model**
```bash
python yolo_pose_training.py --dataset darts_pose_dataset --epochs 100
```



---


## Technology stack

| Component | Choice | Notes |
|-----------|--------|-------|
| Language | **Python 3.12** | tested on 3.12.3 |
| Object detection / segmentation | **Ultralytics YOLO11‑seg** (`8.4.51`) | board, bull, rings, dart masks |
| Keypoint detection | **Ultralytics YOLO11‑pose** | dart tip + tail |
| Deep learning backend | **PyTorch 2.12** (`+cu130`) + torchvision 0.27 | pulled in by Ultralytics; GPU optional |
| Computer vision | **OpenCV** (`opencv-python 4.10.0.84`) | I/O, ellipse fit, homography, drawing |
| Numerics | **NumPy 2.1.3** | PCA endpoints, geometry |
| Config | **PyYAML 6.0.2** | dataset `data.yaml` |
| Annotation | **Roboflow** | instance‑segmentation labelling + YOLO export |

A CUDA GPU is **optional** — everything runs on CPU, just slower. Training realistically wants a GPU (the code defaults target a 4 GB laptop GPU, e.g. an RTX 3050, with `batch=4`).


