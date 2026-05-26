"""Quick visual test for the darts score detector on static images.

Usage:
  python test.py                        # random images — press any key for next, 'q' to quit
  python test.py --image path/to/img   # specific image
  python test.py --all                  # loop through every image in order, press any key to advance
  python test.py --rings                # overlay the scoring rings for visual debugging
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

DRAW_RINGS = False  # toggled by --rings flag

# Import geometry helpers from the main detection script
sys.path.insert(0, str(Path(__file__).parent))
from darts_score_detection_offline import (
    score_geometric,
    label_to_score,
    perspective_correct_vec,
    nms_indices,
    dart_tip_from_endpoints,
    refine_dart_tip,
    BULL_TO_DOUBLE_OUTER,
    BULL_POLYGON_SCALE,
    INNER_BULL_EDGE,
    OUTER_BULL_EDGE,
    TRIPLE_INNER,
    TRIPLE_OUTER,
    DOUBLE_INNER,
    DOUBLE_OUTER,
    iou,
)


def draw_scoring_rings(frame, cx, cy, bull_radius, ema_ellipse, angle_offset):
    """Overlay the regulation scoring rings on the image for visual debugging.
    Rings are drawn in perspective by mapping each radius into the ellipse coords.
    """
    if bull_radius is None or bull_radius <= 0:
        return
    # We draw concentric "rings" by sampling angles around the bull and mapping
    # each point back through the inverse of perspective_correct_vec.
    if ema_ellipse is not None:
        _, _, major, minor, ea = ema_ellipse
        scale = min(major / max(minor, 1.0), 4.0)
        ar = math.radians(ea)
        cos_a, sin_a = math.cos(ar), math.sin(ar)
    else:
        scale = 1.0
        cos_a, sin_a = 1.0, 0.0

    ring_radii = [INNER_BULL_EDGE, OUTER_BULL_EDGE, TRIPLE_INNER, TRIPLE_OUTER, DOUBLE_INNER, DOUBLE_OUTER]
    ring_colors = [(0,255,255), (0,255,255), (255,180,0), (255,180,0), (0,180,255), (0,180,255)]
    for radius_units, color in zip(ring_radii, ring_colors):
        r_pixels = radius_units * bull_radius
        pts = []
        for ang in np.linspace(0, 2 * math.pi, 96):
            # Build the corrected vector at this radius and angle
            dx_c = r_pixels * math.cos(ang)
            dy_c = r_pixels * math.sin(ang)
            # Inverse perspective: undo the minor-axis stretch
            u =  cos_a * dx_c + sin_a * dy_c
            v = -sin_a * dx_c + cos_a * dy_c
            v /= scale
            dx = cos_a * u - sin_a * v
            dy = sin_a * u + cos_a * v
            pts.append((int(cx + dx), int(cy + dy)))
        cv2.polylines(frame, [np.array(pts, dtype=np.int32)], True, color, 1)

    # Draw sector spokes
    for k in range(20):
        # Sector boundary k is at angle k*18 - 9 (centered at k*18) — boundaries
        # at -9, 9, 27, ... in board-rotated frame
        boundary_deg = k * 18.0 - 9.0
        # Convert to image angle: board frame 0 = sector-20 direction = up = -90 in image
        # The score code does: a = atan2(dy,dx) - angle_offset + 99 ;  boundary at a=k*18.
        # → boundary in image space: atan2 = k*18 - 99 + angle_offset
        img_ang = math.radians(k * 18.0 - 99.0 + angle_offset)
        # corrected vector at edge:
        r_pixels = DOUBLE_OUTER * bull_radius
        dx_c = r_pixels * math.cos(img_ang)
        dy_c = r_pixels * math.sin(img_ang)
        u =  cos_a * dx_c + sin_a * dy_c
        v = -sin_a * dx_c + cos_a * dy_c
        v /= scale
        dx = cos_a * u - sin_a * v
        dy = sin_a * u + cos_a * v
        # Start at outer bull
        r_start = OUTER_BULL_EDGE * bull_radius
        dx_s_c = r_start * math.cos(img_ang)
        dy_s_c = r_start * math.sin(img_ang)
        us = cos_a * dx_s_c + sin_a * dy_s_c
        vs = -sin_a * dx_s_c + cos_a * dy_s_c
        vs /= scale
        sdx = cos_a * us - sin_a * vs
        sdy = sin_a * us + cos_a * vs
        cv2.line(frame, (int(cx + sdx), int(cy + sdy)),
                 (int(cx + dx), int(cy + dy)), (80, 80, 80), 1)

VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
DEFAULT_MODEL = 'models/yolo/best.pt'
DEFAULT_CONF  = 0.35


def detect_image(frame, yolo, yolo_pose, dart_cls, bull_cls, board_cls):
    """Run detection on a single BGR frame. Returns annotated frame and list of score labels."""
    h, w = frame.shape[:2]
    # Clean grayscale conversion at the very beginning to prevent overlay contamination
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results = yolo(frame, conf=DEFAULT_CONF, verbose=False)[0]
    seg_polys = results.masks.xy if (results.masks is not None and results.masks.xy is not None) else None

    results_pose = None
    if yolo_pose is not None:
        results_pose = yolo_pose(frame, conf=DEFAULT_CONF, verbose=False)[0]

    # ── Board FIRST — authoritative source for center, radius, perspective ──
    # Derive board geometry
    ema_ellipse, angle_offset = None, 0.0
    board_poly_active = None
    if board_cls is not None:
        best_board_idx, best_board_conf = -1, 0.0
        for i, box in enumerate(results.boxes):
            if int(box.cls[0]) == board_cls:
                conf = float(box.conf[0])
                if conf > best_board_conf:
                    best_board_conf, best_board_idx = conf, i
        if best_board_idx >= 0 and best_board_conf > 0.5 and \
                seg_polys is not None and len(seg_polys[best_board_idx]) >= 5:
            board_poly = seg_polys[best_board_idx].astype(np.float32)
            board_poly_active = board_poly
            pts = board_poly.astype(np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 200, 0))
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.polylines(frame, [pts], True, (0, 200, 0), 2)
            try:
                (ex, ey), (ax1, ax2), ea = cv2.fitEllipse(board_poly)
                major, minor = max(ax1, ax2) / 2.0, min(ax1, ax2) / 2.0
                if ax1 > ax2:
                    ea = (ea + 90.0) % 180.0
                if minor > 10.0:
                    ema_ellipse = (ex, ey, major, minor, ea)
            except cv2.error:
                pass

    # ── Bull center & radius — board-derived when available ─────────────────
    center_x, center_y, bull_radius = w / 2.0, h / 2.0, 19.0  # fallback

    if ema_ellipse is not None:
        # Authoritative: board ellipse derives everything
        bex, bey, bmaj, _, _ = ema_ellipse
        center_x, center_y = bex, bey
        bull_radius = bmaj * BULL_TO_DOUBLE_OUTER

    best_bull_idx, best_bull_conf = -1, 0.0
    for i, box in enumerate(results.boxes):
        if int(box.cls[0]) == bull_cls:
            conf = float(box.conf[0])
            if conf > best_bull_conf:
                best_bull_conf, best_bull_idx = conf, i

    if best_bull_idx >= 0:
        bx1, by1, bx2, by2 = results.boxes[best_bull_idx].xyxy[0].tolist()
        poly = seg_polys[best_bull_idx] if (seg_polys is not None and len(seg_polys[best_bull_idx]) >= 3) else None
        
        if poly is not None:
            pts = poly.astype(np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        cv2.putText(frame, f'Bull {best_bull_conf:.2f}', (int(bx1), int(by1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Get bull center coordinates (high precision local center)
        bull_cx = bull_cy = None
        if poly is not None:
            poly32 = poly.astype(np.float32)
            M = cv2.moments(poly32)
            if M['m00'] > 1e-3:
                bull_cx = M['m10'] / M['m00']
                bull_cy = M['m01'] / M['m00']
        if bull_cx is None:
            bull_cx = (bx1 + bx2) / 2.0
            bull_cy = (by1 + by2) / 2.0
            
        center_x, center_y = bull_cx, bull_cy

        if ema_ellipse is None:
            # No board available — fall back to the bull polygon for radius
            if poly is not None:
                poly32 = poly.astype(np.float32)
                if len(poly) >= 5:
                    try:
                        (ex, ey), (aw, ah), _ = cv2.fitEllipse(poly32)
                        major_b = max(aw, ah) / 2.0
                        minor_b = min(aw, ah) / 2.0
                        # Reject contaminated polygons (darts crossing the bull): extreme aspect ratio
                        if minor_b > 1.0 and major_b / minor_b < 2.0:
                            bull_radius = major_b * BULL_POLYGON_SCALE   # major axis = true diameter
                        else:
                            bull_radius = (min(bx2 - bx1, by2 - by1) / 2.0) * BULL_POLYGON_SCALE
                    except cv2.error:
                        bull_radius = (min(bx2 - bx1, by2 - by1) / 2.0) * BULL_POLYGON_SCALE
                else:
                    bull_radius = (min(bx2 - bx1, by2 - by1) / 2.0) * BULL_POLYGON_SCALE
            else:
                bull_radius = (min(bx2 - bx1, by2 - by1) / 2.0) * BULL_POLYGON_SCALE
                cv2.circle(frame, (int(center_x), int(center_y)), int(bull_radius), (0, 255, 0), 2)

    # ── Angle offset from topmost board vertex cluster ──────────────────────
    if board_poly_active is not None and ema_ellipse is not None:
        n_top = max(3, len(board_poly_active) // 10)
        top_indices = np.argsort(board_poly_active[:, 1])[:n_top]
        raw_dx = float(np.mean(board_poly_active[top_indices, 0])) - center_x
        raw_dy = float(np.mean(board_poly_active[top_indices, 1])) - center_y
        c_dx, c_dy = perspective_correct_vec(raw_dx, raw_dy, ema_ellipse)
        angle_offset = math.degrees(math.atan2(c_dy, c_dx)) + 90.0

    # ── Darts ─────────────────────────────────────────────────────────────────
    # Determine which model is authoritative for darts
    use_pose = (results_pose is not None)
    dart_source = results_pose if use_pose else results
    
    # Class ID for dart in the chosen model
    active_dart_cls = 0 if use_pose else dart_cls

    dart_boxes, dart_confs, dart_orig = [], [], []
    for i, box in enumerate(dart_source.boxes):
        if int(box.cls[0]) == active_dart_cls:
            dart_boxes.append(box.xyxy[0].tolist())
            dart_confs.append(float(box.conf[0]))
            dart_orig.append(i)

    labels = []
    total = 0
    for k in nms_indices(dart_boxes, dart_confs):
        x1, y1, x2, y2 = dart_boxes[k]
        orig_i = dart_orig[k]

        tip_cx, tip_cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        if use_pose:
            # Draw matching segmentation polygon if available
            best_seg_idx = -1
            best_iou = 0.3
            for i_seg, box_seg in enumerate(results.boxes):
                if int(box_seg.cls[0]) == dart_cls:
                    seg_bbox = box_seg.xyxy[0].tolist()
                    curr_iou = iou(dart_boxes[k], seg_bbox)
                    if curr_iou > best_iou:
                        best_iou = curr_iou
                        best_seg_idx = i_seg

            if best_seg_idx >= 0 and seg_polys is not None and len(seg_polys[best_seg_idx]) >= 3:
                pts = seg_polys[best_seg_idx].astype(np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], (0, 165, 255))
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                cv2.polylines(frame, [pts], True, (0, 165, 255), 2)
            else:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)

            # Pose keypoints
            if dart_source.keypoints is not None and dart_source.keypoints.xy is not None:
                kpts = dart_source.keypoints.xy[orig_i].cpu().numpy()
                if len(kpts) >= 2 and not np.allclose(kpts, 0.0):
                    pk_tip, pk_tail = kpts[0], kpts[1]
                    tip_cx, tip_cy = float(pk_tip[0]), float(pk_tip[1])
                    tail_x, tail_y = float(pk_tail[0]), float(pk_tail[1])
                    
                    # Runtime guard: tip must be closer to the bull center than tail
                    dist_tip = (tip_cx - center_x) ** 2 + (tip_cy - center_y) ** 2
                    dist_tail = (tail_x - center_x) ** 2 + (tail_y - center_y) ** 2
                    if dist_tail < dist_tip:
                        tip_cx, tip_cy, tail_x, tail_y = tail_x, tail_y, tip_cx, tip_cy
                        
                    # Refine pose keypoint
                    tip_cx, tip_cy = refine_dart_tip(gray, (tip_cx, tip_cy), (tail_x, tail_y), search_px=18)
                    # Draw pose visualizer (skeleton line)
                    cv2.line(frame, (int(tail_x), int(tail_y)), (int(tip_cx), int(tip_cy)), (0, 255, 255), 2)
                    cv2.circle(frame, (int(tail_x), int(tail_y)), 5, (255, 0, 0), -1) # Tail blue
        else:
            # Fallback to segmentation polygons
            if seg_polys is not None and len(seg_polys[orig_i]) >= 3:
                poly = seg_polys[orig_i].astype(np.float32)
                cx_p = float(np.mean(poly[:, 0]))
                cy_p = float(np.mean(poly[:, 1]))
                pts_c = poly - np.array([cx_p, cy_p], dtype=np.float32)
                _, eigvecs = np.linalg.eigh(np.cov(pts_c.T))
                major = eigvecs[:, -1]
                proj = pts_c @ major
                end_a = poly[int(np.argmin(proj))]
                end_b = poly[int(np.argmax(proj))]
                
                # Choose tip as the endpoint closer to the bullseye
                dist_a = (end_a[0] - center_x) ** 2 + (end_a[1] - center_y) ** 2
                dist_b = (end_b[0] - center_x) ** 2 + (end_b[1] - center_y) ** 2
                if dist_a < dist_b:
                    tip_cx, tip_cy = float(end_a[0]), float(end_a[1])
                    tail_x, tail_y = float(end_b[0]), float(end_b[1])
                else:
                    tip_cx, tip_cy = float(end_b[0]), float(end_b[1])
                    tail_x, tail_y = float(end_a[0]), float(end_a[1])
                    
                tip_cx, tip_cy = refine_dart_tip(
                    gray, (tip_cx, tip_cy), (tail_x, tail_y))
                
                # Draw filled segmentation polygon
                pts = seg_polys[orig_i].astype(np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], (0, 165, 255))
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                cv2.polylines(frame, [pts], True, (0, 165, 255), 2)
            else:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)

        dx, dy = tip_cx - center_x, tip_cy - center_y
        dx_c, dy_c = perspective_correct_vec(dx, dy, ema_ellipse)
        label, _ = score_geometric(center_x, center_y,
                                   center_x + dx_c, center_y + dy_c,
                                   bull_radius, angle_offset)
        labels.append(label)
        total += label_to_score(label)

        # Highlight tip with red dot
        cv2.circle(frame, (int(tip_cx), int(tip_cy)), 4, (0, 0, 255), -1)
        cv2.putText(frame, f"score: {label}", (int(x1) + 5, int(y1) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    if labels:
        cv2.putText(frame, f"total: {total}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    if DRAW_RINGS and center_x is not None:
        draw_scoring_rings(frame, center_x, center_y, bull_radius, ema_ellipse, angle_offset)

    # Draw sector-20 direction indicator
    if ema_ellipse is not None:
        dir_rad = math.radians(angle_offset - 90.0)
        line_len = int(bull_radius * 2.5)
        end_x = int(center_x + line_len * math.cos(dir_rad))
        end_y = int(center_y + line_len * math.sin(dir_rad))
        cv2.line(frame, (int(center_x), int(center_y)), (end_x, end_y), (0, 255, 255), 2)
        cv2.putText(frame, '20', (end_x + 4, end_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return frame, labels, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default='', help='Path to a specific test image')
    ap.add_argument('--all', action='store_true', help='Test every image in data/images/')
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--rings', action='store_true',
                    help='Overlay the regulation scoring rings on the image for visual debugging')
    ap.add_argument('--save', default='', help='Optional folder to save annotated images instead of showing them')
    args = ap.parse_args()

    global DRAW_RINGS
    DRAW_RINGS = args.rings

    if not Path(args.model).is_file():
        sys.exit(f"ERROR: model not found: {args.model}")

    yolo = YOLO(args.model)
    name_to_id = {v: k for k, v in yolo.names.items()}
    bull_cls = name_to_id.get('bull', name_to_id.get('bulls'))
    dart_cls = name_to_id.get('dart', name_to_id.get('darts',
               name_to_id.get('arrow', name_to_id.get('arrows'))))
    board_cls = name_to_id.get('board', name_to_id.get('boards'))
    if bull_cls is None or dart_cls is None:
        sys.exit(f"ERROR: model must have bull and dart classes. Found: {list(yolo.names.values())}")

    # Load Pose model if available for hybrid tracking
    yolo_pose = None
    pose_path = Path(args.model).parent / 'pose_best.pt'
    if not pose_path.is_file():
        pose_path = Path('models/yolo/pose_best.pt')
    if pose_path.is_file():
        yolo_pose = YOLO(str(pose_path))
        print(f"Hybrid mode active: using YOLOv11 Pose model {pose_path} for precise dart tip detection.")
    else:
        print("Segmentation mode active (no pose model found). Using segmentation polygons.")

    images_dir = Path('data') / 'images'
    all_imgs = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in VALID_EXTS)

    if args.image:
        image_paths = [Path(args.image)]
    else:
        if not all_imgs:
            sys.exit(f"No images found in {images_dir}")
        image_paths = all_imgs

    print(f"Model: {args.model}  |  classes: bull={bull_cls} dart={dart_cls} board={board_cls}")
    if args.save:
        print(f"Saving annotations for {len(image_paths)} image(s) to folder: {args.save}\n")
    else:
        print(f"Testing {len(image_paths)} image(s) — press Left/Right Arrow to navigate, 'W' to toggle annotations, 'Q' to quit\n")

    save_dir = Path(args.save) if args.save else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    if save_dir is not None:
        for path in image_paths:
            frame = cv2.imread(str(path))
            if frame is None:
                print(f"  SKIP (unreadable): {path}")
                continue
            out, labels, total = detect_image(frame, yolo, yolo_pose, dart_cls, bull_cls, board_cls)
            print(f"  {path.name:40s}  darts={labels}  total={total}")
            cv2.imwrite(str(save_dir / path.name), out)
    else:
        idx = 0
        show_annotations = True
        last_idx = -1
        last_show_annotations = None

        while True:
            if not image_paths:
                print("No images to test.")
                break

            path = image_paths[idx]
            frame = cv2.imread(str(path))
            if frame is None:
                print(f"  SKIP (unreadable): {path}")
                image_paths.pop(idx)
                if idx >= len(image_paths):
                    idx = 0
                continue

            if show_annotations:
                annotated_frame = frame.copy()
                out, labels, total = detect_image(annotated_frame, yolo, yolo_pose, dart_cls, bull_cls, board_cls)
                if idx != last_idx or show_annotations != last_show_annotations:
                    print(f"  {path.name:40s}  darts={labels}  total={total}")
                cv2.imshow('Darts Test', out)
            else:
                if idx != last_idx or show_annotations != last_show_annotations:
                    print(f"  {path.name:40s}  [Annotations Hidden]")
                cv2.imshow('Darts Test', frame)

            last_idx = idx
            last_show_annotations = show_annotations

            k = cv2.waitKey(0)
            k_code = k & 0xFF

            if k_code == ord('q') or k == 27:
                break
            elif k_code == ord('w') or k_code == ord('W'):
                show_annotations = not show_annotations
            elif k == 81 or k == 65361 or k == 2424832 or k_code == 81:
                idx = (idx - 1) % len(image_paths)
            elif k == 83 or k == 65363 or k == 2424834 or k_code == 83:
                idx = (idx + 1) % len(image_paths)

    if save_dir is None:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
