# 🎯 Darts Score Detector

Automatically reads a dartboard from a camera, webcam, RTSP stream, video file, or
still image and **scores every dart** in standard darts notation — `S20`, `T17`,
`D5`, `B25` (outer bull), `B50` (inner bull), or `Miss`.

The scoring is **fully deterministic geometry**: there is no "score classifier"
neural net. Two YOLO models only *locate* things on the board (the bull, the
rings, and each dart tip); a regulation‑dartboard model then converts every dart
tip's `(distance, angle)` relative to the bull into a score with a lookup table.
That makes the result interpretable and easy to debug.

---

## Table of contents

1. [How it works (architecture)](#how-it-works-architecture)
2. [Technology stack](#technology-stack)
3. [Installation](#installation)
4. [Running it / "playing"](#running-it--playing)
5. [The dartboard scoring model](#the-dartboard-scoring-model)
6. [Annotating images in Roboflow](#annotating-images-in-roboflow)
7. [Re-training the models](#re-training-the-models)
8. [Project layout](#project-layout)
9. [Main scripts reference](#main-scripts-reference)
10. [scratch/ — developer debug scripts](#scratch--developer-debug-scripts)
11. [Troubleshooting](#troubleshooting)
12. [License](#license)

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

**Geometric robustness.** A camera never sees the board perfectly face‑on, so the
board appears as an ellipse. The detector fits that ellipse, builds a **projective
homography** that rectifies the board plane to a true circle, and measures every
dart in that rectified metric space. This is what keeps the thin double/triple
bands at the rim accurate. (An affine fallback is used if the homography is
rejected as implausible.) Bull centre, radius, board ellipse, and rotation are
**EMA‑smoothed across frames** for stability on video, and per‑dart scores use a
**temporal majority vote** over a short track.

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

A CUDA GPU is **optional** — everything runs on CPU, just slower. Training
realistically wants a GPU (the code defaults target a 4 GB laptop GPU, e.g. an
RTX 3050, with `batch=4`).

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

`requirements.txt` pins `ultralytics`, `opencv-python`, `numpy`, and `pyyaml`.
Installing `ultralytics` automatically pulls a compatible `torch`/`torchvision`.
For an explicit CUDA build, install the matching torch wheel from
<https://pytorch.org> **before** `pip install -r requirements.txt`.

The two trained model weights (`models/yolo/best.pt` and `models/yolo/pose_best.pt`)
**ship with the repository**, so inference works immediately after install — no
training required to try it.

---

## Running it / "playing"

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

**On‑screen display.** Each dart is boxed and labelled with its score
(`score: T20`); the running round total is shown top‑left (`total: 60`); the
detected bull, 50‑bull, and double‑20 arc are boxed with confidences; the "20"
sector direction is drawn as a line.

**Controls (live/video window):**
| Key | Action |
|-----|--------|
| `q` | quit |
| `SPACE` | (with `--capture`) save the current **raw** frame to `data/images/` for re‑annotation |

**Useful flags:**
| Flag | Default | Purpose |
|------|---------|---------|
| `--yolo-model` | `models/yolo/best.pt` | segmentation weights |
| `--yolo-pose` | `models/yolo/pose_best.pt` | pose (tip) weights; falls back to seg polygons if missing |
| `--conf` | `0.35` | YOLO confidence threshold |
| `--angle-offset DEG` | `0.0` | manually rotate the scoring frame so "20" is at top |
| `--disable-auto-rotation` | off | stop auto‑aligning to the detected double‑20 arc |
| `--output FILE.mp4` | — | write annotated video |
| `--capture` | off | enable `SPACE` to save raw frames into `--capture-dir` |
| `--capture-dir DIR` | `data/images` | where captured frames go |
| `--raw-view` | off | show the unannotated feed |

> **Tip — capturing training data.** Run with `--capture` on your real camera,
> press `SPACE` whenever the board/darts look good, and the raw frames land in
> `data/images/`. Upload those to Roboflow to grow your dataset.

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

## The dartboard scoring model

All distances are expressed in **OuterBull‑radius units** (the outer‑bull radius =
15.9 mm = `1.0`). Regulation radii (from `darts_score_detection_offline.py`):

| Zone | Radius range (bull‑radius units) | Result |
|------|----------------------------------|--------|
| Inner bull (B50) | `r < 0.399` | `B50` = 50 |
| Outer bull (B25) | `0.399 ≤ r ≤ 1.0` | `B25` = 25 |
| Single (inner) | `1.0 < r < 6.226` | `S<n>` = n |
| Triple | `6.226 ≤ r ≤ 6.730` | `T<n>` = 3·n |
| Single (outer) | `6.730 < r < 10.189` | `S<n>` = n |
| Double | `10.189 ≤ r ≤ 10.692` | `D<n>` = 2·n |
| Outside | `r > 10.692` | `Miss` = 0 |

The sector number `n` comes from the angle, walking clockwise from "20" at top:

```
SECTORS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
```

Each sector spans 18°. The board's true rotation is detected automatically from
the **double‑20 arc** class; `--angle-offset` overrides/assists this if needed.

---

## Annotating images in Roboflow

The segmentation model is trained on **instance‑segmentation polygons** exported
from [Roboflow](https://roboflow.com).

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

There are **two** models. Retrain the segmentation model whenever you change
classes or add board imagery; regenerate + retrain the pose model whenever the
segmentation labels change (the pose labels are derived from them).

### A. Segmentation model (board, bull, rings, dart)

```bash
python yolo_training.py --dataset darts_dataset --epochs 150
```

What it does:
1. Reads class names from the dataset's `data.yaml`.
2. Auto‑splits `train/` into `train`/`valid`/`test` (defaults 15 % / 10 %) if those
   splits are missing.
3. Rewrites a normalized `data.yaml` with absolute paths.
4. Trains YOLO11‑seg with augmentations tuned for ~300–500 images.
5. Copies the best checkpoint to **`models/yolo/best.pt`** (where inference looks).

Common overrides: `--model yolo11s-seg.pt` (larger/more accurate),
`--epochs 300`, `--imgsz 1024`, `--batch 8`, `--device 0` (force GPU 0),
`--device cpu`.

### B. Pose model (dart tip + tail) — two steps

**Step 1 — convert the segmentation dataset to a pose dataset.** This derives a
2‑keypoint (tip, tail) label for every `arrow` polygon, using PCA major‑axis
endpoints with a width cue (the wider end is the flight/tail, the narrower end is
the tip — the same routine inference trusts):

```bash
python convert_seg_to_pose.py --src darts_dataset --dst darts_pose_dataset
```

**Step 2 — train the pose model:**

```bash
python yolo_pose_training.py --dataset darts_pose_dataset --epochs 100
```

It trains YOLO11‑pose at `imgsz=960` (dart tips are tiny; high resolution matters)
with perspective augmentation, and copies the best checkpoint to
**`models/yolo/pose_best.pt`**.

> ⚠️ **Always regenerate the pose dataset (Step 1) after re‑annotating**, then
> retrain (Step 2). The pose model's tip accuracy is capped by its labels, and the
> labels come from the segmentation polygons.

### Base checkpoints

Training starts from the standard Ultralytics base checkpoints
(`yolo11n-seg.pt`, `yolo11n-pose.pt`). These are gitignored and download
automatically on first use; you can also drop them in the repo root.

---

## Project layout

```
Darts Score/
├── darts_score_detection_offline.py   # ▶ main inference entry point (live/video/image/RTSP)
├── test.py                            # still-image smoke test + ring overlay
├── yolo_training.py                   # train YOLO11-seg  → models/yolo/best.pt
├── convert_seg_to_pose.py             # seg dataset → 2-keypoint pose dataset
├── yolo_pose_training.py              # train YOLO11-pose → models/yolo/pose_best.pt
├── requirements.txt
├── LICENSE                            # MIT
├── README.md
├── models/yolo/
│   ├── best.pt                        # trained segmentation weights  (tracked)
│   └── pose_best.pt                   # trained pose weights          (tracked)
├── darts_dataset/                     # Roboflow seg export            (gitignored)
├── darts_pose_dataset/                # converted pose dataset         (gitignored)
├── data/images/                       # captured frames                (gitignored)
├── runs/                              # YOLO training outputs           (gitignored)
└── scratch/                           # one-off debug scripts (gitignored; see below)
```

---

## Main scripts reference

| Script | Purpose | Key functions |
|--------|---------|---------------|
| `darts_score_detection_offline.py` | Inference & scoring on any source. | `score_geometric` (zone lookup), `board_rectifier`/`rect_offset`/`rect_to_image` (homography), `perspective_correct_vec` (affine fallback), `dart_tip_tail_poly` + `refine_dart_tip` (tip from polygon), `resolve_class`, `draw_scoring_rings`, `Track`/`nms_indices` (temporal vote), `process()` (per‑frame loop). |
| `test.py` | Visual smoke test on still images; imports the geometry helpers above so the test path matches production exactly. | `detect_image`, `draw_scoring_rings`. |
| `yolo_training.py` | Train the segmentation model. | `ensure_split`, `write_normalized_yaml`. |
| `convert_seg_to_pose.py` | Turn seg polygons into pose keypoints. | `dart_tip_tail` (PCA + width tip/tail), `dart_tip_from_endpoints`, `resolve_dart_class`. |
| `yolo_pose_training.py` | Train the pose model. | `main` (auto‑adds `flip_idx`). |

---

## scratch/ — developer debug scripts

`scratch/` holds throwaway, single‑purpose scripts written while debugging the
geometry and dataset. **They are gitignored** (kept locally, not published) and
are not needed to run, train, or annotate. They are documented here so the intent
behind each is preserved.

> ⚠️ Many hardcode local image paths / old class assumptions. Treat them as a lab
> notebook, not maintained tooling.

**Scoring / geometry verification**
| Script | What it checks |
|--------|----------------|
| `check_calculation.py` | Replicates `test.py`'s geometry to verify a score by hand. |
| `analyze_images.py` | Prints the computed ring radii (`TRIPLE_INNER`, `TRIPLE_OUTER`, …). |
| `test_homography_scoring.py` | Synthetic validation of the projective rectifier vs the affine model (tilted‑board accuracy). |
| `test_decision_logic.py` | Exercises the bull‑centre choice logic on flagged large‑shift cases. |
| `find_doubles.py` | Lists images where a dart lands in a double ring, with its radius ratio. |
| `find_s5.py` | Finds darts scored in sector 5 (angle/rotation sanity check). |
| `count_swaps.py` | Counts how often tip/tail disambiguation flips vs the old method. |

**Bull / board centre + alignment**
| Script | What it checks |
|--------|----------------|
| `check_centers.py` / `check_centers_all.py` | Compares board‑ vs bull‑derived centres across images. |
| `check_alignment.py` / `debug_alignment.py` | Inspects detection boxes and board/bull alignment. |
| `locate_physical_bull.py` / `locate_physical_bull_794298.py` | Locates the true physical bull on specific frames. |
| `inspect_bull_poly.py` | Dumps the bull polygon/box and confidence. |
| `analyze_board_poly.py` | Reports the board polygon vertex count/shape. |

**Ring / ratio measurement (constant calibration)**
| Script | What it checks |
|--------|----------------|
| `measure_dataset_ratios.py` / `check_dataset_ratios.py` | Measures bull/triple/double radius ratios across the dataset. |
| `check_board_bull_ratio.py` | Per‑image board‑major / bull‑major radius ratio. |
| `measure_physical_rings.py` / `measure_one_image.py` | Measures physical ring radii on one or many images. |

**Mask / moments / polygon experiments**
| Script | What it checks |
|--------|----------------|
| `test_moments.py` / `test_real_moments.py` / `test_mask_moments.py` / `test_mask_moments_all.py` | Compare `cv2.moments` centroids vs polygon means for the bull centre. |
| `analyze_gradient.py` | Inspects the intensity gradient used by `refine_dart_tip`. |
| `test_refinement.py` / `test_refinement_correct_orientation.py` | Validate `refine_dart_tip` snapping/no‑drift behaviour. |

**Dart tip / pose**
| Script | What it checks |
|--------|----------------|
| `test_pose.py` | Runs the pose model and prints predicted boxes/keypoints. |
| `verify_dart_positions.py` | Prints each dart's predicted tip coordinates. |
| `check_dart3.py` | Focused check on a particular dart/bull‑centre case. |

**Detection / image inspection**
| Script | What it checks |
|--------|----------------|
| `inspect_detections.py` | Dumps raw detections for a few `data/images/` frames. |
| `inspect_image_darts.py` | Loads an image and inspects detected darts. |
| `test_single_image.py` | End‑to‑end detection on one image. |
| `find_dots.py` | Detects bull/centre dots on a frame. |
| `list_red_regions.py` | Lists red regions (double/triple ring colour) in an image. |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `YOLO weights missing: models/yolo/best.pt` | The weights ship with the repo; if absent, train with `yolo_training.py`. |
| `Failed to open input` on a webcam | The script prints available `/dev/video*` indices — pass the right `--input N`. |
| Sector numbers are rotated/wrong | The board roll isn't auto‑detected — set `--angle-offset DEG` (or check the `twenty` class is being detected). |
| Tip lands on the wrong side of a double/triple line | Retrain the pose model after regenerating `darts_pose_dataset` (Step B above). |
| Slow on CPU | Expected; use a CUDA GPU and pass `--device 0`. |
| `darts_dataset/` got committed | It's gitignored now; if previously tracked, run `git rm -r --cached darts_dataset`. |

---

## License

[MIT](LICENSE) © 2026 Rares Gherasa.

Dartboard geometry constants follow regulation board dimensions. YOLO models are
trained with [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL‑3.0 —
review their license terms if you redistribute trained weights commercially).
