"""Train a YOLO11 instance-segmentation model on a Roboflow-exported dataset.

Workflow:
  1. Point --dataset at the unzipped Roboflow folder (must contain `train/images`
     and `train/labels`). If `valid/` or `test/` are missing, they are split off
     from `train/` automatically (default 80/10/10).
  2. A normalized data.yaml is written next to the dataset with absolute paths.
  3. Ultralytics YOLO11-seg is trained with augmentations tuned for the small
     dart dataset (~300 images).
  4. The best weights are copied to `models/yolo/best.pt` so the offline
     detector picks them up by default.

Example:
  python yolo_training.py --dataset "yolov11 dataset" --model yolo11n-seg.pt --epochs 150
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def ensure_split(dataset_dir: Path, val_frac: float, test_frac: float, seed: int):
    """Move a random fraction of train images+labels into valid/ and test/ if missing."""
    train_imgs = dataset_dir / 'train' / 'images'
    train_lbls = dataset_dir / 'train' / 'labels'
    if not train_imgs.is_dir() or not train_lbls.is_dir():
        sys.exit(f"Missing train/images or train/labels under {dataset_dir}")

    have_valid = (dataset_dir / 'valid' / 'images').is_dir() and any(
        (dataset_dir / 'valid' / 'images').iterdir())
    have_test = (dataset_dir / 'test' / 'images').is_dir() and any(
        (dataset_dir / 'test' / 'images').iterdir())
    if have_valid and have_test:
        print(f"Found existing valid/ and test/ splits — skipping auto-split")
        return

    images = sorted(p for p in train_imgs.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    rng = random.Random(seed)
    rng.shuffle(images)
    n_total = len(images)
    n_val = max(1, int(n_total * val_frac)) if not have_valid else 0
    n_test = max(1, int(n_total * test_frac)) if not have_test else 0

    val_set = images[:n_val]
    test_set = images[n_val:n_val + n_test]

    def move_pair(img_path: Path, dst_root: Path):
        (dst_root / 'images').mkdir(parents=True, exist_ok=True)
        (dst_root / 'labels').mkdir(parents=True, exist_ok=True)
        lbl_path = train_lbls / (img_path.stem + '.txt')
        shutil.move(str(img_path), str(dst_root / 'images' / img_path.name))
        if lbl_path.exists():
            shutil.move(str(lbl_path), str(dst_root / 'labels' / lbl_path.name))

    if not have_valid:
        for p in val_set:
            move_pair(p, dataset_dir / 'valid')
        print(f"Split {len(val_set)} images into valid/")
    if not have_test:
        for p in test_set:
            move_pair(p, dataset_dir / 'test')
        print(f"Split {len(test_set)} images into test/")


def write_normalized_yaml(dataset_dir: Path, names: list[str]) -> Path:
    """Write a data.yaml next to the dataset with absolute paths and correct class names."""
    out = dataset_dir / 'data.yaml'
    cfg = {
        'path': str(dataset_dir.resolve()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(names),
        'names': names,
    }
    with open(out, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Wrote {out}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='darts_dataset',
                   help="Path to unzipped Roboflow dataset")
    p.add_argument('--model', default='yolo11n-seg.pt',
                   help="Base model (yolo11n-seg.pt / yolo11s-seg.pt / yolo11m-seg.pt)")
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=4,
                   help="Batch size — keep ≤4 for 4 GB VRAM (RTX 3050 Laptop)")
    p.add_argument('--workers', type=int, default=4,
                   help="DataLoader workers — high values waste RAM/CPU on laptops")
    p.add_argument('--device', default='',
                   help="'' = auto (GPU if available), '0' = first GPU, 'cpu' = force CPU")
    p.add_argument('--val-frac', type=float, default=0.15)
    p.add_argument('--test-frac', type=float, default=0.10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--project', default='runs',
                   help="Output project root. YOLO prepends 'segment/' so this yields runs/segment/<name>/.")
    p.add_argument('--name', default='darts_yolo11')
    p.add_argument('--output-weights', default='models/yolo/best.pt',
                   help="Where to copy the best.pt after training")
    args = p.parse_args()

    dataset_dir = Path(args.dataset).resolve()
    if not dataset_dir.is_dir():
        sys.exit(f"Dataset folder not found: {dataset_dir}")

    # ── Read class names from the existing Roboflow data.yaml if present ─────
    existing_yaml = dataset_dir / 'data.yaml'
    names = ['arrow', 'board', 'bulls']  # fallback matches v2 Roboflow export
    if existing_yaml.is_file():
        with open(existing_yaml) as f:
            data = yaml.safe_load(f)
        if isinstance(data.get('names'), list):
            names = data['names']
        elif isinstance(data.get('names'), dict):
            names = [data['names'][k] for k in sorted(data['names'])]
    print(f"Class names: {names}")

    # ── Split + rewrite data.yaml ────────────────────────────────────────────
    ensure_split(dataset_dir, args.val_frac, args.test_frac, args.seed)
    data_yaml = write_normalized_yaml(dataset_dir, names)

    # ── Train ────────────────────────────────────────────────────────────────
    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
        # Augmentations tuned for ~300 images:
        mosaic=1.0,
        close_mosaic=10,       # disable mosaic last 10 epochs for clean fine-tuning
        mixup=0.10,
        copy_paste=0.30,       # only meaningful for segmentation, helps small data
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,          # small rotation — board is mostly fixed
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        fliplr=0.5,            # safe: YOLO only locates dart+bull, not zone
        flipud=0.0,
        erasing=0.2,
        # Optimizer:
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
        patience=30,           # early-stop if no val improvement for 30 epochs
        plots=True,
    )

    # ── Copy best.pt to the location the offline script expects ─────────────
    save_dir = Path(results.save_dir) if hasattr(results, 'save_dir') else \
               Path(args.project) / args.name
    best = save_dir / 'weights' / 'best.pt'
    if best.is_file():
        out = Path(args.output_weights)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, out)
        print(f"\n✓ Copied {best} → {out}")
        print(f"Run inference with: python darts_score_detection_offline.py --yolo-model {out}")
    else:
        print(f"\n⚠ best.pt not found at {best}", file=sys.stderr)


if __name__ == '__main__':
    main()
