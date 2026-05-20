# Darts Score Detection

Detects darts and the bullseye in a live camera or RTSP stream and scores each dart's landing zone (`S20`, `T17`, `D5`, `B25`, `B50`, …).

**Architecture:** YOLO11 instance segmentation (Ultralytics) locates `bulls` and `dart`. Each dart's tip is taken from the segmentation polygon, then a regulation dartboard geometry (ring radii in OuterBull-radius units + 20-sector lookup) maps `(angle, distance)` → score. No classifier model; scoring is fully deterministic.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Run

```bash
python darts_score_detection_offline.py --input 0
# or:
python darts_score_detection_offline.py --input path/to/video.mp4
python darts_score_detection_offline.py --input path/to/image.jpg
python darts_score_detection_offline.py --input "rtsp://user:pass@host:554/stream"
```

Flags:
- `--yolo-model models/yolo/best.pt` — YOLO weights (default points here)
- `--conf 0.35` — YOLO confidence threshold
- `--angle-offset DEG` — rotate scoring frame so the "20" sector aligns with the top of the image. Set this if the board is rolled in the camera view.
- `--output out.mp4` — write annotated video
- `--capture` — `SPACE` saves a raw frame to `data/images/` for re-annotation

Press `q` in the window to quit.

---

## Train the YOLO detector

1. Annotate in **Roboflow** (instance segmentation, two classes: `bulls` and `dart`). Export as **YOLOv11** and unzip into the repo root as `yolov11 dataset/`.

2. Train:

   ```bash
   python yolo_training.py --dataset "yolov11 dataset" --epochs 150
   ```

   The script auto-splits `train/` into `train/valid/test` if `valid/` and `test/` are missing, rewrites `data.yaml` with absolute paths, trains with augmentations tuned for ~300 images, and copies the best weights to `models/yolo/best.pt`.

   Common overrides: `--model yolo11s-seg.pt` (larger, more accurate), `--epochs 300`, `--imgsz 1024`, `--device 0` (force GPU 0), `--batch 8`.

---

## Layout

```
darts_score_detection_offline.py   inference entry point
yolo_training.py                   YOLO11-seg training
test.py                            random-image smoke test
models/yolo/best.pt                trained YOLO weights (created by yolo_training.py)
data/images/                       captured frames
yolov11 dataset/                   Roboflow YOLOv11 export
runs/                              YOLO training outputs (auto-created)
yolo11n-seg.pt                     base YOLO11-seg weights for training
```
