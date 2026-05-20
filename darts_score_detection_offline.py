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
        return f"T{sector_num}", 3 * sector_num
    if DOUBLE_INNER <= r <= DOUBLE_OUTER:
        return f"D{sector_num}", 2 * sector_num
    return f"S{sector_num}", sector_num


# ── Module-level helpers ──────────────────────────────────────────────────────
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
    p.add_argument('--input', default='0',
                   help="Webcam index (e.g. '0'), RTSP URL, or path to video/image file")
    p.add_argument('--yolo-model', default='models/yolo/best.pt',
                   help="Path to YOLO11/YOLOv8 weights (.pt)")
    p.add_argument('--conf', type=float, default=0.35,
                   help="YOLO confidence threshold")
    p.add_argument('--output', default='',
                   help="Optional output video path (e.g. out.mp4)")
    p.add_argument('--capture', action='store_true',
                   help="Enable SPACE-to-save screenshots of the raw (un-annotated) frame")
    p.add_argument('--capture-dir', default='data/images',
                   help="Folder where screenshots are saved when --capture is set")
    p.add_argument('--angle-offset', type=float, default=0.0,
                   help="Degrees added to dart angle so the '20' sector aligns with the top of the image. "
                        "If your board is rolled in the camera frame, set this to compensate.")
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

    # Resolve class indices by name so class order in data.yaml doesn't matter.
    # Accept 'bull' or 'bulls' (Roboflow exports with plural by default).
    name_to_id = {v: k for k, v in yolo.names.items()}
    BULL_CLS = name_to_id.get('bull', name_to_id.get('bulls'))
    DART_CLS = name_to_id.get('dart', name_to_id.get('darts'))
    if BULL_CLS is None or DART_CLS is None:
        print(f"ERROR: model must have a bull and a dart class. Found: {list(yolo.names.values())}", file=sys.stderr)
        sys.exit(1)

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

    def process(frame, use_tracker=True):
        nonlocal writer, writer_w, writer_h
        nonlocal ema_cx, ema_cy, ema_r
        h, w = frame.shape[:2]

        results = yolo(frame, conf=args.conf, verbose=False)[0]

        seg_polys = None
        if results.masks is not None and results.masks.xy is not None:
            seg_polys = results.masks.xy  # list of (N,2) arrays, one per box

        # ── Bull detection ─────────────────────────────────────────────────
        best_bull_idx = -1
        best_bull_conf = 0.0
        for i, box in enumerate(results.boxes):
            if int(box.cls[0]) != BULL_CLS:
                continue
            conf = float(box.conf[0])
            if conf > best_bull_conf:
                best_bull_conf = conf
                best_bull_idx = i

        if best_bull_idx >= 0:
            bx1, by1, bx2, by2 = results.boxes[best_bull_idx].xyxy[0].tolist()
            if seg_polys is not None and len(seg_polys[best_bull_idx]) >= 3:
                poly = seg_polys[best_bull_idx]
                M = cv2.moments(poly.astype(np.float32))
                if M['m00'] > 1e-3:
                    new_cx = M['m10'] / M['m00']
                    new_cy = M['m01'] / M['m00']
                else:
                    new_cx, new_cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                # ACC-3: use bbox radius (geometry-consistent with the regulation
                # mm-ratio constants above, which assume OuterBull radius == bbox half-side).
                new_r = max(bx2 - bx1, by2 - by1) / 2.0
                cv2.polylines(frame, [poly.astype(np.int32)], True, (0, 255, 0), 2)
            else:
                new_cx, new_cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                new_r = max(bx2 - bx1, by2 - by1) / 2.0
                cv2.circle(frame, (int(new_cx), int(new_cy)), int(new_r), (0, 255, 0), 2)
            cv2.putText(frame, f'Bull {best_bull_conf:.2f}', (int(bx1), int(by1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ACC-9: EMA only when conf > 0.6 AND center stable (or first detection)
            if best_bull_conf > 0.6:
                if ema_cx is None:
                    ema_cx, ema_cy, ema_r = new_cx, new_cy, new_r
                else:
                    if math.hypot(new_cx - ema_cx, new_cy - ema_cy) < 50.0:
                        ema_cx = 0.85 * ema_cx + 0.15 * new_cx
                        ema_cy = 0.85 * ema_cy + 0.15 * new_cy
                        ema_r  = 0.85 * ema_r  + 0.15 * new_r

        if ema_cx is not None:
            center_x, center_y, bull_radius = ema_cx, ema_cy, ema_r
        else:
            center_x, center_y, bull_radius = 0.0, 0.0, bull_radius_fallback

        # ── Dart detections ────────────────────────────────────────────────
        dart_boxes = []
        dart_confs = []
        dart_orig_idx = []
        for i, box in enumerate(results.boxes):
            if int(box.cls[0]) != DART_CLS:
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

            # ACC-1: dart tip = polygon vertex farthest from bull center
            tip_cx = (x1 + x2) / 2
            tip_cy = (y1 + y2) / 2
            if seg_polys is not None and len(seg_polys[orig_i]) >= 3:
                poly = seg_polys[orig_i]
                d2 = (poly[:, 0] - center_x) ** 2 + (poly[:, 1] - center_y) ** 2
                far = int(np.argmax(d2))
                tip_cx = float(poly[far, 0])
                tip_cy = float(poly[far, 1])

            label, score = score_geometric(center_x, center_y, tip_cx, tip_cy,
                                           bull_radius, args.angle_offset)
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'label': label,
                'score': score,
                'tip': (tip_cx, tip_cy),
                'orig_idx': orig_i,
            })

        # ACC-8: temporal smoothing via IoU tracking (majority vote per track)
        display_items = []  # (bbox, label, low_conf, tip, orig_idx)
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
                    display_items.append((det['bbox'], top, low_conf, det['tip'], det['orig_idx']))
                else:
                    new_tr = Track(det['bbox'], det['label'], current_frame)
                    tracks.append(new_tr)
                    display_items.append((det['bbox'], det['label'], False, det['tip'], det['orig_idx']))
            tracks[:] = [tr for tr in tracks
                         if current_frame - tr.last_seen_frame <= TRACK_MAX_AGE]
            frame_idx[0] = current_frame + 1
        else:
            for det in detections:
                display_items.append((det['bbox'], det['label'], False, det['tip'], det['orig_idx']))

        # ── Draw dart annotations ──────────────────────────────────────────
        total_round_score = 0
        for bbox, label, low_conf, tip, orig_i in display_items:
            x1, y1, x2, y2 = bbox
            display_label = label + ("?" if low_conf else "")
            text_color = (0, 165, 255) if low_conf else (255, 255, 255)

            if seg_polys is not None and len(seg_polys[orig_i]) >= 3:
                cv2.polylines(frame, [seg_polys[orig_i].astype(np.int32)], True, (0, 165, 255), 2)
            else:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)

            # ACC-1: draw dart tip
            cv2.circle(frame, (int(tip[0]), int(tip[1])), 4, (0, 0, 255), -1)

            cv2.putText(frame, f"score: {display_label}", (int(x1) + 5, int(y1) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)

            # Sum the integer scores for a HUD total
            _, score_int = score_geometric(center_x, center_y, tip[0], tip[1],
                                           bull_radius, args.angle_offset)
            total_round_score += score_int

        if display_items:
            cv2.putText(frame, f"total: {total_round_score}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

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
        out = process(frames[0], use_tracker=False)
        cv2.imshow('Darts Score Detection', out)
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
            if args.capture:
                cv2.putText(out, f"SPACE=save  q=quit  saved={saved}",
                            (10, out.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('Darts Score Detection', out)
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
