"""Convert a YOLO segmentation dataset into a 2-keypoint Pose dataset.

For every dart/arrow instance the polygon's two long-axis endpoints are emitted
as keypoints (kp0 = tip, kp1 = tail), where the tip is the more acute endpoint.
All other classes (board, bull rings, sector-20) are dropped — the Pose model
only localises dart tips.

The dart class is resolved by name from the source `data.yaml`, so this works
for both the old export ('arrow','board','bulls') and the new Darts.yolo26
export ('The 25/50-point circle','arrow','dartboard','twenty').

Example:
  python convert_seg_to_pose.py --src Darts.yolo26 --dst darts_pose_dataset
"""
import argparse
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


def dart_tip_from_endpoints(poly, end_a, end_b):
    n = len(poly)
    k = max(2, n // 12)
    def _angle(ep):
        idx = int(np.argmin((poly[:, 0] - ep[0]) ** 2 + (poly[:, 1] - ep[1]) ** 2))
        v1 = poly[(idx - k) % n] - poly[idx]
        v2 = poly[(idx + k) % n] - poly[idx]
        d1, d2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if d1 < 1e-6 or d2 < 1e-6:
            return math.pi
        return math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2)) / (d1 * d2))))
    return end_a if _angle(end_a) <= _angle(end_b) else end_b


def dart_tip_tail(poly):
    """Return (tip, tail) px points for a dart polygon.

    Endpoints come from the PCA major axis (robust to curved/L-shaped masks,
    unlike farthest-pair-from-centroid which can pick two points on the same
    end). The tip is then disambiguated by *width*: a dart's flight/tail is
    markedly wider than the barrel that tapers to the tip, so the narrower end
    is the tip. Vertex acuteness breaks ties when the two ends are similar.
    """
    c = poly.mean(axis=0)
    pts_c = poly - c
    try:
        evecs = np.linalg.eigh(np.cov(pts_c.T))[1]
    except np.linalg.LinAlgError:
        # degenerate; fall back to farthest-pair
        d_c = (pts_c[:, 0]) ** 2 + (pts_c[:, 1]) ** 2
        end_a = poly[int(np.argmax(d_c))]
        d_a = (poly[:, 0] - end_a[0]) ** 2 + (poly[:, 1] - end_a[1]) ** 2
        end_b = poly[int(np.argmax(d_a))]
        tip = dart_tip_from_endpoints(poly, end_a, end_b)
        return tip, (end_b if np.allclose(tip, end_a) else end_a)

    major, minor = evecs[:, -1], evecs[:, 0]
    proj_maj = pts_c @ major
    proj_min = pts_c @ minor
    end_a = poly[int(np.argmin(proj_maj))]
    end_b = poly[int(np.argmax(proj_maj))]

    L = float(proj_maj.max() - proj_maj.min())
    if L < 1e-6:
        tip = dart_tip_from_endpoints(poly, end_a, end_b)
        return tip, (end_b if np.allclose(tip, end_a) else end_a)

    # Width of each end = minor-axis spread of the points in its outer 25% band.
    band = 0.25 * L
    a_band = proj_min[proj_maj <= proj_maj.min() + band]
    b_band = proj_min[proj_maj >= proj_maj.max() - band]
    width_a = float(a_band.max() - a_band.min()) if a_band.size >= 2 else 0.0
    width_b = float(b_band.max() - b_band.min()) if b_band.size >= 2 else 0.0

    # Clear width difference (>15%) is decisive; else fall back to acuteness.
    if abs(width_a - width_b) > 0.15 * max(width_a, width_b, 1e-6):
        if width_a < width_b:
            return end_a, end_b
        return end_b, end_a
    tip = dart_tip_from_endpoints(poly, end_a, end_b)
    return tip, (end_b if np.allclose(tip, end_a) else end_a)


def resolve_dart_class(data_yaml: Path) -> int:
    """Return the class id of the dart/arrow class from a Roboflow data.yaml."""
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    names = data.get('names')
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if not isinstance(names, list):
        sys.exit(f"Could not read class names from {data_yaml}")
    for cid, name in enumerate(names):
        if name.lower() in ('dart', 'darts', 'arrow', 'arrows'):
            return cid
    sys.exit(f"No dart/arrow class found in {data_yaml} (names={names})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='darts_dataset',
                    help="Path to the segmentation dataset (must contain data.yaml)")
    ap.add_argument('--dst', default='darts_pose_dataset',
                    help="Output path for the converted pose dataset")
    ap.add_argument('--val-split', type=float, default=0.1,
                    help="Fraction of train images to reserve for validation (default 0.1). "
                         "Used when the source dataset has no valid/ split.")
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    src_dir = (Path(__file__).parent / args.src).resolve() if not Path(args.src).is_absolute() else Path(args.src)
    dst_dir = (Path(__file__).parent / args.dst).resolve() if not Path(args.dst).is_absolute() else Path(args.dst)

    if not src_dir.is_dir():
        sys.exit(f"Source dataset directory not found: {src_dir}")
    data_yaml = src_dir / 'data.yaml'
    if not data_yaml.is_file():
        sys.exit(f"data.yaml not found in {src_dir}")

    dart_cls = resolve_dart_class(data_yaml)
    print(f"Converting {src_dir} -> {dst_dir}  (dart class id = {dart_cls})")

    # When the source has no valid split, carve one off from train.
    val_override: set[str] = set()
    if args.val_split > 0:
        src_train_lbl = src_dir / 'train' / 'labels'
        src_valid_lbl = src_dir / 'valid' / 'labels'
        has_valid = src_valid_lbl.is_dir() and any(src_valid_lbl.glob('*.txt'))
        if not has_valid and src_train_lbl.is_dir():
            import random
            rng = random.Random(args.seed)
            all_stems = [p.stem for p in src_train_lbl.glob('*.txt')]
            rng.shuffle(all_stems)
            n_val = max(1, int(len(all_stems) * args.val_split))
            val_override = set(all_stems[:n_val])
            print(f"No valid split found — reserving {n_val}/{len(all_stems)} train images for validation.")

    for split in ['train', 'valid', 'test']:
        (dst_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (dst_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    for split in ['train', 'valid', 'test']:
        src_lbl_dir = src_dir / split / 'labels'
        src_img_dir = src_dir / split / 'images'
        if not src_lbl_dir.is_dir() or not src_img_dir.is_dir():
            print(f"Skipping split '{split}' (missing images/ or labels/).")
            continue

        label_files = list(src_lbl_dir.glob('*.txt'))
        print(f"Processing split '{split}': {len(label_files)} label files...")

        for lbl_path in label_files:
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                cand = src_img_dir / (lbl_path.stem + ext)
                if cand.exists():
                    img_path = cand
                    break
            if img_path is None:
                print(f"Warning: no image for label {lbl_path.name}")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Error: could not read image {img_path}")
                continue
            h, w = img.shape[:2]

            dst_split = 'valid' if (split == 'train' and lbl_path.stem in val_override) else split
            shutil.copy(str(img_path), str(dst_dir / dst_split / 'images' / img_path.name))

            pose_lines = []
            with open(lbl_path) as f:
                lines = f.readlines()

            for line in lines:
                parts = line.split()
                if not parts or int(parts[0]) != dart_cls:
                    continue

                poly_norm = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
                poly_px = poly_norm * [w, h]
                if len(poly_px) < 3:
                    continue

                # PCA major-axis endpoints + width-based tip/tail disambiguation
                # (matches the inference path; far more robust than farthest-pair).
                tip, tail = dart_tip_tail(poly_px)
                tip_norm = tip / [w, h]
                tail_norm = tail / [w, h]

                xmin, xmax = poly_norm[:, 0].min(), poly_norm[:, 0].max()
                ymin, ymax = poly_norm[:, 1].min(), poly_norm[:, 1].max()
                box_cx, box_cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
                box_w, box_h = xmax - xmin, ymax - ymin

                # class x y w h  tip_x tip_y vis  tail_x tail_y vis  (class id 0 = dart)
                pose_lines.append(
                    f"0 {box_cx:.6f} {box_cy:.6f} {box_w:.6f} {box_h:.6f} "
                    f"{tip_norm[0]:.6f} {tip_norm[1]:.6f} 2 "
                    f"{tail_norm[0]:.6f} {tail_norm[1]:.6f} 2\n")

            with open(dst_dir / dst_split / 'labels' / lbl_path.name, 'w') as f:
                f.writelines(pose_lines)

    yaml_cfg = {
        'path': str(dst_dir.resolve()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'kpt_shape': [2, 3],  # 2 keypoints (tip, tail), each (x, y, visibility)
        'flip_idx': [0, 1],  # 2 keypoints (tip, tail) map directly to themselves under horizontal flip
        'nc': 1,
        'names': ['dart'],
    }
    with open(dst_dir / 'data.yaml', 'w') as f:
        yaml.safe_dump(yaml_cfg, f, sort_keys=False)
    print(f"Wrote {dst_dir / 'data.yaml'}")
    print("Conversion completed successfully!")


if __name__ == '__main__':
    main()
