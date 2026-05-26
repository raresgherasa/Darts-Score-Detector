"""Offline darts score detector — YOLO11 for detection, deterministic geometric scoring.

The DNN classifier was removed (ACC-2): a regulation dartboard's score zones are
fully determined by (distance, angle) relative to the bull, so a lookup table is
more reliable and interpretable than a 60-class MLP, and removes the
features.tsv → training → ONNX dependency.

Requires:  pip install ultralytics

Example:
  python darts_score_detection_offline.py --input 0
  python darts_score_detection_offline.py --input path/to/video.mp4
  python darts_score_detection_offline.py --input path/to/image.jpg
  python darts_score_detection_offline.py --input rtsp://user:pass@ip/stream
  python darts_score_detection_offline.py --input 1 --capture
"""

import argparse
import math
import os
import pathlib
import sys
import time
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO


# ── Dartboard geometry (regulation board, distances in OuterBull-radius units) ─
# OuterBull diameter = 31.8 mm → radius 15.9 mm is our unit.
OUTER_BULL_MM   = 15.9
DOUBLE_OUTER_MM = 170.0
# The board polygon detected by YOLO includes the outer black numbers ring,
# which puts the physical double outer edge at approximately 0.824 * board_major_radius.
# Thus, BULL_TO_DOUBLE_OUTER relative to the board polygon major axis is:
# 0.824 * (OUTER_BULL_MM / DOUBLE_OUTER_MM) = 0.824 * 0.09353 = 0.07707
BULL_TO_DOUBLE_OUTER = 0.824 * (OUTER_BULL_MM / DOUBLE_OUTER_MM) # ~0.07707

# The detected bull polygon is about 1.23x larger than the physical outer bull (15.9 mm).
# To scale the detected bull radius down to the physical outer bull radius:
BULL_POLYGON_SCALE = 1.0 / 1.23   # ~0.813

INNER_BULL_EDGE = 6.35 / 15.9          # 0.399  — B50 inside this
OUTER_BULL_EDGE = 1.0                  #         — B25 up to here
TRIPLE_INNER    = 99.0 / 15.9          # 6.226
TRIPLE_OUTER    = 107.0 / 15.9         # 6.730
DOUBLE_INNER    = 162.0 / 15.9         # 10.189
DOUBLE_OUTER    = 170.0 / 15.9         # 10.692

# Sector numbers walking clockwise from "20" at the top of the board.
SECTORS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]


def score_geometric(cx, cy, tip_x, tip_y, bull_radius, angle_offset_deg=0.0):
    """Return (label_str, score_int) from board geometry.

    cx, cy        — bull center in image pixels
    tip_x, tip_y  — dart tip in image pixels
    bull_radius   — OuterBull radius in image pixels
    angle_offset_deg — rotate so the "20" sector is at the top of the image
    """
    if bull_radius <= 0:
        return "?", 0
    dx = tip_x - cx
    dy = tip_y - cy
    r = math.hypot(dx, dy) / bull_radius

    if r < INNER_BULL_EDGE:
        return "B50", 50
    if r <= OUTER_BULL_EDGE:
        return "B25", 25
    if r > DOUBLE_OUTER:
        return "Miss", 0

    # Angle relative to image-right axis. Image y grows downward; "20" is at
    # top-center which is dy<0 → atan2 returns ~-90°. We shift so that the "20"
    # sector starts at 0° in the rotated frame.
    angle_deg = math.degrees(math.atan2(dy, dx)) - angle_offset_deg
    a = (angle_deg + 99.0) % 360.0   # +99° = +90° (move 20-center to 0) + 9° (half sector)
    sector_idx = int(a // 18.0) % 20
    sector_num = SECTORS[sector_idx]

    if TRIPLE_INNER <= r <= TRIPLE_OUTER:
        return f"D{sector_num}", 2 * sector_num
    if DOUBLE_INNER <= r <= DOUBLE_OUTER:
        return f"T{sector_num}", 3 * sector_num
    return f"S{sector_num}", sector_num


# ── Module-level helpers ──────────────────────────────────────────────────────
def label_to_score(label: str) -> int:
    """Convert a score label ('T20', 'D10', 'S5', 'B50', 'B25', 'Miss') to its integer value."""
    if label == 'B50':
        return 50
    if label == 'B25':
        return 25
    if len(label) >= 2 and label[0] in ('T', 'D', 'S'):
        try:
            n = int(label[1:])
            return {'T': 3, 'D': 2, 'S': 1}[label[0]] * n
        except ValueError:
            pass
    return 0


def iou(boxA, boxB):
    """IoU between two boxes given as (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    areaA = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    areaB = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = areaA + areaB - inter
    if union <= 0:
        return 0.0
    return inter / union


def nms_indices(boxes, confs, iou_thresh=0.5):
    """Return indices of boxes kept after greedy NMS (sorted by conf desc)."""
    order = sorted(range(len(boxes)), key=lambda i: confs[i], reverse=True)
    kept = []
    for idx in order:
        if any(iou(boxes[idx], boxes[k]) > iou_thresh for k in kept):
            continue
        kept.append(idx)
    return kept


def dart_tip_from_endpoints(poly, end_a, end_b):
    """Return the dart tip as the more acute of the two polygon endpoints.

    Uses interior angle at each endpoint — dart tips come to a sharp point,
    flights/tails are wider. Robust to board position and camera angle.
    """
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


def refine_dart_tip(gray, approx_tip, tail_pt, search_px=18, strip_half=4, grad_thresh=None):
    """Refine approximate dart tip along the shaft using gradient magnitude profile.

    Scans outwards from the tail/approx_tip and finds where the shaft gradient drops.
    Falls back to approx_tip when geometry is degenerate.
    """
    tip = np.array(approx_tip, dtype=np.float64)
    tail = np.array(tail_pt, dtype=np.float64)
    shaft = tip - tail
    shaft_len = float(np.linalg.norm(shaft))
    if shaft_len < 5.0:
        return approx_tip
    shaft_dir = shaft / shaft_len
    perp_dir = np.array([-shaft_dir[1], shaft_dir[0]])

    h, w = gray.shape[:2]
    # Scan a bit deeper inside the shaft and further outside
    margin = search_px + strip_half + 6
    x0 = max(0, int(tip[0]) - margin)
    y0 = max(0, int(tip[1]) - margin)
    x1 = min(w, int(tip[0]) + margin + 1)
    y1 = min(h, int(tip[1]) + margin + 1)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return approx_tip
    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    def _peak(center_pt):
        best = 0.0
        for s in range(-strip_half, strip_half + 1):
            pt = center_pt + s * perp_dir
            xi = int(round(pt[0])) - x0
            yi = int(round(pt[1])) - y0
            if 0 <= xi < grad.shape[1] and 0 <= yi < grad.shape[0]:
                v = float(grad[yi, xi])
                if v > best:
                    best = v
        return best

    # Sample gradient profile along the shaft
    t_vals = list(range(-10, search_px + 1))
    profile = []
    for t in t_vals:
        profile.append(_peak(tip + t * shaft_dir))
        
    profile = np.array(profile)
    
    # Define baseline shaft gradient inside the dart body
    inside_indices = [i for i, t in enumerate(t_vals) if t <= 2]
    max_inside_grad = np.max(profile[inside_indices]) if len(inside_indices) > 0 else 100.0
    thresh = max(40.0, 0.35 * max_inside_grad)
    
    best_t = 0
    for i, t in enumerate(t_vals):
        if t < 0:
            continue
        if profile[i] >= thresh:
            best_t = t
        else:
            if i + 1 < len(t_vals) and profile[i+1] < thresh:
                break
                
    refined = tip + best_t * shaft_dir
    return (float(refined[0]), float(refined[1]))


def perspective_correct_vec(dx, dy, ema_ellipse):
    """Apply fitEllipse-derived affine de-foreshortening to a (dx, dy) vector.

    Stretches the minor-axis component by major/minor to restore the circular
    board geometry distorted by a non-perpendicular camera.
    Returns corrected (dx, dy). If ema_ellipse is None or degenerate, returns input unchanged.
    """
    if ema_ellipse is None:
        return dx, dy
    _ex, _ey, major, minor, ea = ema_ellipse
    if minor < 1.0 or major < minor:
        return dx, dy
    scale = min(major / minor, 4.0)
    angle_rad = math.radians(ea)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    u =  cos_a * dx + sin_a * dy   # component along major axis (unchanged)
    v = -sin_a * dx + cos_a * dy   # component along minor axis (compressed)
    v *= scale
    return cos_a * u - sin_a * v, sin_a * u + cos_a * v


class Track:
    """Per-dart temporal track that votes on a stable label across frames."""
    __slots__ = ('bbox', 'votes', 'last_seen_frame')

    def __init__(self, bbox, label, last_seen_frame):
        self.bbox = bbox
        self.votes = Counter([label])
        self.last_seen_frame = last_seen_frame

    def push(self, label):
        self.votes[label] += 1

    def majority(self):
        top, top_count = self.votes.most_common(1)[0]
        total = sum(self.votes.values())
        return top, top_count, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='rtsp://user:pass@host:554/stream',
                   help="Webcam index (e.g. '0'), RTSP URL, or path to video/image file")
    p.add_argument('--yolo-model', default='models/yolo/best.pt',
                   help="Path to YOLO11/YOLOv8 weights (.pt)")
    p.add_argument('--yolo-pose', default='models/yolo/pose_best.pt',
                   help="Path to YOLO11 Pose weights (.pt). If present, enables hybrid tracking.")
    p.add_argument('--conf', type=float, default=0.35,
                   help="YOLO confidence threshold")
    p.add_argument('--output', default='',
                   help="Optional output video path (e.g. out.mp4)")
    p.add_argument('--capture', action='store_true',
                   help="Enable SPACE-to-save screenshots of the raw (un-annotated) frame")
    p.add_argument('--capture-dir', default='data/images',
                   help="Folder where screenshots are saved when --capture is set")
    p.add_argument('--raw-view', action='store_true',
                   help="Show the raw camera feed in the display window without overlays")
    p.add_argument('--angle-offset', type=float, default=0.0,
                   help="Fallback rotation (degrees) used only when the model has no 'board' class. "
                        "When 'board' is detected, rotation is computed automatically from the board polygon.")
    args = p.parse_args()

    # ── INF-3: verify YOLO weights exist before doing anything ──────────────
    if not pathlib.Path(args.yolo_model).is_file():
        print(f"ERROR: YOLO weights missing: {args.yolo_model} — run yolo_training.py to produce them.",
              file=sys.stderr)
        sys.exit(1)

    # ── YOLO detector ───────────────────────────────────────────────────────
    yolo = YOLO(args.yolo_model)
    print(f"YOLO model loaded: {args.yolo_model}")
    print(f"Class names: {yolo.names}")

    # Load Pose model if available for hybrid tracking
    yolo_pose = None
    if pathlib.Path(args.yolo_pose).is_file():
        yolo_pose = YOLO(args.yolo_pose)
        print(f"YOLO Pose model loaded: {args.yolo_pose} (Hybrid mode active)")
    else:
        pose_fallback = pathlib.Path(args.yolo_model).parent / 'pose_best.pt'
        if pose_fallback.is_file():
            yolo_pose = YOLO(str(pose_fallback))
            print(f"YOLO Pose model loaded: {pose_fallback} (Hybrid mode active)")
        else:
            print("INFO: No YOLO Pose weights found. Falling back to segmentation polygons.")

    # Resolve class indices by name so class order in data.yaml doesn't matter.
    # Accept 'bull' or 'bulls' (Roboflow exports with plural by default).
    name_to_id = {v: k for k, v in yolo.names.items()}
    BULL_CLS = name_to_id.get('bull', name_to_id.get('bulls'))
    DART_CLS = name_to_id.get('dart', name_to_id.get('darts',
               name_to_id.get('arrow', name_to_id.get('arrows'))))
    if BULL_CLS is None or DART_CLS is None:
        print(f"ERROR: model must have a bull and a dart class. Found: {list(yolo.names.values())}", file=sys.stderr)
        sys.exit(1)
    BOARD_CLS = name_to_id.get('board', name_to_id.get('boards'))
    if BOARD_CLS is None:
        print("INFO: model has no 'board' class — perspective correction and auto-rotation disabled.")
    else:
        print(f"INFO: 'board' class found (id={BOARD_CLS}) — perspective correction and auto-rotation enabled.")

    # ── Input source ────────────────────────────────────────────────────────
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    is_image = os.path.isfile(args.input) and args.input.lower().endswith(image_exts)
    is_rtsp = args.input.lower().startswith(('rtsp://', 'rtsps://'))

    if is_image:
        frames = [cv2.imread(args.input)]
        if frames[0] is None:
            print(f"Failed to read image: {args.input}", file=sys.stderr)
            sys.exit(1)
        cap = None
    else:
        if args.input.isdigit():
            src = int(args.input)
            cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(src)
        elif is_rtsp:
            cap = cv2.VideoCapture(args.input, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"Failed to open input: {args.input}", file=sys.stderr)
            if not is_rtsp:
                # BUG-3: explicit loop to avoid leaking VideoCapture handles
                available = []
                for i in range(4):
                    probe = cv2.VideoCapture(i, cv2.CAP_V4L2)
                    if probe.isOpened():
                        available.append(i)
                    probe.release()
                print("Available devices: " + " ".join(str(i) for i in available),
                      file=sys.stderr)
            sys.exit(1)
        if is_rtsp:
            print(f"RTSP stream opened: {args.input}")
        frames = None

    writer = None
    writer_w = None
    writer_h = None
    # ACC-9: EMA-smoothed bull center/radius across frames
    ema_cx = None
    ema_cy = None
    ema_r = None
    bull_radius_fallback = 19.0  # fallback until YOLO detects the bull

    # ACC-8: temporal majority vote per dart track (only used for video)
    tracks = []
    frame_idx = [0]
    TRACK_IOU_THRESH = 0.4
    TRACK_MAX_AGE = 30
    ema_ellipse = None        # (ex, ey, major, minor, ea) — board ellipse EMA
    ema_angle_offset = None   # auto-detected board rotation (deg); None → use --angle-offset

    def process(frame, use_tracker=True):
        nonlocal writer, writer_w, writer_h
        nonlocal ema_cx, ema_cy, ema_r
        nonlocal ema_ellipse, ema_angle_offset
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = yolo(frame, conf=args.conf, verbose=False)[0]
        results_pose = None
        if yolo_pose is not None:
            results_pose = yolo_pose(frame, conf=args.conf, verbose=False)[0]

        seg_polys = None
        if results.masks is not None and results.masks.xy is not None:
            seg_polys = results.masks.xy  # list of (N,2) arrays, one per box

        # ── Board detection FIRST — authoritative source for center, radius, perspective ─
        # The board polygon is large and immune to dart occlusion. The bull polygon
        # often gets contaminated by overlapping dart shafts, so we derive bull
        # center + radius from the board ellipse via the regulation ratio
        # (OuterBull 15.9 mm / Double-outer 170 mm = 0.0935).
        board_poly_active = None
        if BOARD_CLS is not None:
            best_board_idx = -1
            best_board_conf = 0.0
            for i, box in enumerate(results.boxes):
                if int(box.cls[0]) != BOARD_CLS:
                    continue
                conf = float(box.conf[0])
                if conf > best_board_conf:
                    best_board_conf = conf
                    best_board_idx = i

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
                    # fitEllipse returns (width, height); the MAJOR axis is the true
                    # diameter (perpendicular to perspective tilt) — keep its angle.
                    major = max(ax1, ax2) / 2.0
                    minor = min(ax1, ax2) / 2.0
                    if ax1 > ax2:   # width > height: major axis is horizontal
                        ea = (ea + 90.0) % 180.0
                    if minor > 10.0:
                        if ema_ellipse is None:
                            ema_ellipse = (ex, ey, major, minor, ea)
                        else:
                            oex, oey, omaj, omin, oea = ema_ellipse
                            da = ((ea - oea + 90.0) % 180.0) - 90.0
                            ema_ellipse = (
                                0.85 * oex  + 0.15 * ex,
                                0.85 * oey  + 0.15 * ey,
                                0.85 * omaj + 0.15 * major,
                                0.85 * omin + 0.15 * minor,
                                oea + 0.15 * da,
                            )
                except cv2.error:
                    pass

        # ── Bull center & radius ────────────────────────────────────────────
        # Preferred path: derive from board ellipse (robust to dart occlusion).
        # Fallback path: trust the bull polygon when (a) no board class in the
        # model, or (b) board confidence too low. Even then, prefer the MAJOR
        # axis of the bull ellipse — the major axis equals the true diameter
        # (foreshortening only shrinks the minor axis).
        best_bull_idx = -1
        best_bull_conf = 0.0
        for i, box in enumerate(results.boxes):
            if int(box.cls[0]) != BULL_CLS:
                continue
            conf = float(box.conf[0])
            if conf > best_bull_conf:
                best_bull_conf = conf
                best_bull_idx = i

        new_cx = new_cy = new_r = None

        # 1. Establish scaling and initial center using board ellipse if available
        if ema_ellipse is not None:
            bex, bey, bmaj, _, _ = ema_ellipse
            new_cx, new_cy = bex, bey
            new_r = bmaj * BULL_TO_DOUBLE_OUTER

        # 2. Get the bull center if bull is detected (high precision local center)
        if best_bull_idx >= 0:
            bx1, by1, bx2, by2 = results.boxes[best_bull_idx].xyxy[0].tolist()
            cv2.putText(frame, f'Bull {best_bull_conf:.2f}', (int(bx1), int(by1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw bull mask overlay
            if seg_polys is not None and len(seg_polys[best_bull_idx]) >= 3:
                pts = seg_polys[best_bull_idx].astype(np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 0))
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

            # Determine bull center coordinates
            bull_cx = bull_cy = None
            if seg_polys is not None and len(seg_polys[best_bull_idx]) >= 3:
                poly = seg_polys[best_bull_idx].astype(np.float32)
                M = cv2.moments(poly)
                if M['m00'] > 1e-3:
                    bull_cx = M['m10'] / M['m00']
                    bull_cy = M['m01'] / M['m00']
            if bull_cx is None:
                bull_cx = (bx1 + bx2) / 2.0
                bull_cy = (by1 + by2) / 2.0

            # Override the center with the high-precision direct bull detection
            new_cx, new_cy = bull_cx, bull_cy

            # If board is not available, calculate the radius from the bull itself
            if new_r is None:
                if seg_polys is not None and len(seg_polys[best_bull_idx]) >= 5:
                    try:
                        poly = seg_polys[best_bull_idx].astype(np.float32)
                        (ex, ey), (aw, ah), _ = cv2.fitEllipse(poly)
                        major_b = max(aw, ah) / 2.0
                        minor_b = min(aw, ah) / 2.0
                        if minor_b > 1.0 and major_b / minor_b < 2.0:
                            new_r = major_b * BULL_POLYGON_SCALE
                        else:
                            new_r = (min(bx2 - bx1, by2 - by1) / 2.0) * BULL_POLYGON_SCALE
                    except cv2.error:
                        new_r = (min(bx2 - bx1, by2 - by1) / 2.0) * BULL_POLYGON_SCALE
                else:
                    new_r = (min(bx2 - bx1, by2 - by1) / 2.0) * BULL_POLYGON_SCALE

        if new_cx is not None:
            # ACC-9: EMA smoothing (gated on bull conf when bull provides the values;
            # always update when board provides them since the board is very reliable).
            board_drove = (ema_ellipse is not None)
            if board_drove or best_bull_conf > 0.6:
                if ema_cx is None:
                    ema_cx, ema_cy, ema_r = new_cx, new_cy, new_r
                elif math.hypot(new_cx - ema_cx, new_cy - ema_cy) < 50.0:
                    ema_cx = 0.85 * ema_cx + 0.15 * new_cx
                    ema_cy = 0.85 * ema_cy + 0.15 * new_cy
                    ema_r  = 0.85 * ema_r  + 0.15 * new_r

        if ema_cx is not None:
            center_x, center_y, bull_radius = ema_cx, ema_cy, ema_r
        else:
            center_x, center_y, bull_radius = 0.0, 0.0, bull_radius_fallback

        # ── Auto-rotation: sector-20 direction from topmost board vertex ─────
        if board_poly_active is not None and ema_ellipse is not None:
            n_top = max(3, len(board_poly_active) // 10)
            top_indices = np.argsort(board_poly_active[:, 1])[:n_top]
            raw_dx = float(np.mean(board_poly_active[top_indices, 0])) - center_x
            raw_dy = float(np.mean(board_poly_active[top_indices, 1])) - center_y
            c_dx, c_dy = perspective_correct_vec(raw_dx, raw_dy, ema_ellipse)
            new_ao = math.degrees(math.atan2(c_dy, c_dx)) + 90.0
            if ema_angle_offset is None:
                ema_angle_offset = new_ao
            else:
                da2 = ((new_ao - ema_angle_offset + 180.0) % 360.0) - 180.0
                ema_angle_offset = ema_angle_offset + 0.15 * da2

        # active_angle: auto when available, CLI fallback otherwise
        active_angle = ema_angle_offset if ema_angle_offset is not None else args.angle_offset

        # ── Dart detections ────────────────────────────────────────────────
        # Determine which model is authoritative for darts
        use_pose = (results_pose is not None)
        dart_source = results_pose if use_pose else results
        
        # Class ID for dart in the chosen model
        active_dart_cls = 0 if use_pose else DART_CLS

        dart_boxes = []
        dart_confs = []
        dart_orig_idx = []
        for i, box in enumerate(dart_source.boxes):
            if int(box.cls[0]) != active_dart_cls:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            dart_boxes.append((x1, y1, x2, y2))
            dart_confs.append(conf)
            dart_orig_idx.append(i)

        # ACC-12: NMS to dedupe overlapping dart boxes
        keep = nms_indices(dart_boxes, dart_confs, iou_thresh=0.5)

        detections = []
        for k in keep:
            x1, y1, x2, y2 = dart_boxes[k]
            orig_i = dart_orig_idx[k]

            tip_cx = (x1 + x2) / 2
            tip_cy = (y1 + y2) / 2
            
            if use_pose:
                # Pose keypoints
                tail_cx, tail_cy = None, None
                if dart_source.keypoints is not None and dart_source.keypoints.xy is not None:
                    kpts = dart_source.keypoints.xy[orig_i].cpu().numpy()
                    if len(kpts) >= 2 and not np.allclose(kpts, 0.0):
                        pk_tip, pk_tail = kpts[0], kpts[1]
                        tip_cx, tip_cy = float(pk_tip[0]), float(pk_tip[1])
                        tail_cx, tail_cy = float(pk_tail[0]), float(pk_tail[1])
                        
                        # Runtime guard: tip must be closer to the bullseye than tail
                        dist_tip = (tip_cx - center_x) ** 2 + (tip_cy - center_y) ** 2
                        dist_tail = (tail_cx - center_x) ** 2 + (tail_cy - center_y) ** 2
                        if dist_tail < dist_tip:
                            tip_cx, tip_cy, tail_cx, tail_cy = tail_cx, tail_cy, tip_cx, tip_cy
                            
                        # Refine pose keypoint
                        tip_cx, tip_cy = refine_dart_tip(gray, (tip_cx, tip_cy), (tail_cx, tail_cy), search_px=18)
            else:
                # Fallback to segmentation polygons
                tail_cx, tail_cy = None, None
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
                        tail_cx, tail_cy = float(end_b[0]), float(end_b[1])
                    else:
                        tip_cx, tip_cy = float(end_b[0]), float(end_b[1])
                        tail_cx, tail_cy = float(end_a[0]), float(end_a[1])
                        
                    tip_cx, tip_cy = refine_dart_tip(
                        gray, (tip_cx, tip_cy), (tail_cx, tail_cy))

            tip_dx, tip_dy = tip_cx - center_x, tip_cy - center_y
            tip_dx_c, tip_dy_c = perspective_correct_vec(tip_dx, tip_dy, ema_ellipse)
            tip_cx_s = center_x + tip_dx_c
            tip_cy_s = center_y + tip_dy_c
            label, score = score_geometric(center_x, center_y, tip_cx_s, tip_cy_s,
                                           bull_radius, active_angle)
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'label': label,
                'score': score,
                'tip': (tip_cx, tip_cy),
                'tail': (tail_cx, tail_cy) if (tail_cx is not None) else None,
                'orig_idx': orig_i,
            })

        # ACC-8: temporal smoothing via IoU tracking (majority vote per track)
        display_items = []  # (bbox, label, low_conf, tip, tail, orig_idx)
        if use_tracker:
            current_frame = frame_idx[0]
            used_tracks = set()
            for det in detections:
                best_t = -1
                best_iou = TRACK_IOU_THRESH
                for ti, tr in enumerate(tracks):
                    if ti in used_tracks:
                        continue
                    iv = iou(det['bbox'], tr.bbox)
                    if iv > best_iou:
                        best_iou = iv
                        best_t = ti
                if best_t >= 0:
                    tr = tracks[best_t]
                    tr.bbox = det['bbox']
                    tr.push(det['label'])
                    tr.last_seen_frame = current_frame
                    used_tracks.add(best_t)
                    top, top_count, total = tr.majority()
                    low_conf = (total >= 3) and (top_count / total < 0.6)
                    display_items.append((det['bbox'], top, low_conf, det['tip'], det['tail'], det['orig_idx']))
                else:
                    new_tr = Track(det['bbox'], det['label'], current_frame)
                    tracks.append(new_tr)
                    display_items.append((det['bbox'], det['label'], False, det['tip'], det['tail'], det['orig_idx']))
            tracks[:] = [tr for tr in tracks
                         if current_frame - tr.last_seen_frame <= TRACK_MAX_AGE]
            frame_idx[0] = current_frame + 1
        else:
            for det in detections:
                display_items.append((det['bbox'], det['label'], False, det['tip'], det['tail'], det['orig_idx']))

        # ── Draw dart annotations ──────────────────────────────────────────
        total_round_score = 0
        for bbox, label, low_conf, tip, tail, orig_i in display_items:
            x1, y1, x2, y2 = bbox
            display_label = label + ("?" if low_conf else "")
            text_color = (0, 165, 255) if low_conf else (255, 255, 255)

            if tail is not None:
                # Draw skeleton line and tail dot
                cv2.line(frame, (int(tail[0]), int(tail[1])), (int(tip[0]), int(tip[1])), (0, 255, 255), 2)
                cv2.circle(frame, (int(tail[0]), int(tail[1])), 5, (255, 0, 0), -1)
            elif seg_polys is not None and len(seg_polys[orig_i]) >= 3:
                pts = seg_polys[orig_i].astype(np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], (0, 165, 255))
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                cv2.polylines(frame, [pts], True, (0, 165, 255), 2)
            else:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)

            # ACC-1: draw dart tip
            cv2.circle(frame, (int(tip[0]), int(tip[1])), 4, (0, 0, 255), -1)

            cv2.putText(frame, f"score: {display_label}", (int(x1) + 5, int(y1) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)

            total_round_score += label_to_score(label)

        if display_items:
            cv2.putText(frame, f"total: {total_round_score}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Sector-20 direction indicator (cyan) when auto-rotation is active
        if ema_angle_offset is not None and ema_cx is not None:
            dir_rad = math.radians(ema_angle_offset - 90.0)
            line_len = int(bull_radius * 2.5)
            end_x = int(ema_cx + line_len * math.cos(dir_rad))
            end_y = int(ema_cy + line_len * math.sin(dir_rad))
            cv2.line(frame, (int(ema_cx), int(ema_cy)), (end_x, end_y), (0, 255, 255), 2)
            cv2.putText(frame, '20', (end_x + 4, end_y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # ── VideoWriter handling with size-change recovery (QA-4) ─────────
        if args.output:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(args.output, fourcc, 20.0, (w, h))
                writer_w, writer_h = w, h
            elif (w, h) != (writer_w, writer_h):
                print(f"WARNING: frame size changed from ({writer_w},{writer_h}) to "
                      f"({w},{h}); recreating VideoWriter.", file=sys.stderr)
                writer.release()
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(args.output, fourcc, 20.0, (w, h))
                writer_w, writer_h = w, h
            writer.write(frame)
        return frame

    # ── Run ─────────────────────────────────────────────────────────────────
    if is_image:
        raw = frames[0].copy()
        out = process(frames[0], use_tracker=False)
        display_frame = raw if args.raw_view else out
        cv2.imshow('Darts Score Detection', display_frame)
        print("Press any key in the window to exit.")
        cv2.waitKey(0)
    else:
        if args.capture:
            os.makedirs(args.capture_dir, exist_ok=True)
            print(f"Capture mode on — press SPACE to save a raw frame to {args.capture_dir}")
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_rtsp:
                    print("RTSP stream lost — reconnecting…", file=sys.stderr)
                    cap.release()
                    time.sleep(2)
                    cap = cv2.VideoCapture(args.input, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    continue
                break
            raw = frame.copy()
            out = process(frame, use_tracker=True)
            display_frame = raw if args.raw_view else out
            if args.capture:
                # Copy to avoid drawing status text on the clean 'raw' image
                display_frame = display_frame.copy()
                cv2.putText(display_frame, f"SPACE=save  q=quit  saved={saved}",
                            (10, display_frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('Darts Score Detection', display_frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if args.capture and k == 32:  # SPACE
                name = f"darts_{int(time.time() * 1000)}.jpg"
                path = os.path.join(args.capture_dir, name)
                cv2.imwrite(path, raw)
                saved += 1
                print(f"[capture #{saved}] saved {path}")
        cap.release()

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
