import argparse
import shutil
import sys
from pathlib import Path
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='yolov11_pose_dataset',
                   help="Path to converted pose dataset")
    p.add_argument('--model', default='yolo11n-pose.pt',
                   help="Base model (yolo11n-pose.pt / yolo11s-pose.pt)")
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=4,
                   help="Batch size — keep low for small GPU memory")
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='',
                   help="'' = auto, '0' = GPU 0, 'cpu' = CPU")
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--project', default='runs',
                   help="Output project root")
    p.add_argument('--name', default='darts_yolo11_pose')
    p.add_argument('--output-weights', default='models/yolo/pose_best.pt',
                   help="Where to save the best model weights")
    args = p.parse_args()

    dataset_dir = Path(args.dataset).resolve()
    data_yaml = dataset_dir / 'data.yaml'
    if not data_yaml.is_file():
        sys.exit(f"dataset config not found at {data_yaml}")

    print(f"Initializing YOLO Pose training with base model '{args.model}'...")
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
        mosaic=1.0,
        close_mosaic=10,
        mixup=0.10,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        erasing=0.2,
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
        patience=30,
        plots=True,
    )

    save_dir = Path(results.save_dir) if hasattr(results, 'save_dir') else \
               Path(args.project) / 'pose' / args.name
    best = save_dir / 'weights' / 'best.pt'
    if best.is_file():
        out = Path(args.output_weights)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, out)
        print(f"\n✓ Copied best weights: {best} → {out}")
    else:
        print(f"\n⚠ best.pt not found at {best}", file=sys.stderr)

if __name__ == '__main__':
    main()
