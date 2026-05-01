"""Train a YOLOv8 model on a custom YOLO-format dataset.

Example:
    python src/train.py --data configs/data.yaml --model yolov8n.pt \\
        --epochs 50 --imgsz 640 --batch 16
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 detector")
    p.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    p.add_argument("--model", type=str, default="yolov8n.pt",
                   help="Base model: yolov8n/s/m/l/x.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="",
                   help="cuda device, e.g. 0 or 0,1 or cpu (auto if empty)")
    p.add_argument("--project", type=str, default="runs/detect")
    p.add_argument("--name", type=str, default="train")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[INFO] Training {args.model} on {args.data}")
    print(f"[INFO] epochs={args.epochs} imgsz={args.imgsz} batch={args.batch}")

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        plots=True,
        verbose=True,
    )

    save_dir = Path(results.save_dir)
    print(f"\n[OK] Training complete. Weights at: {save_dir / 'weights' / 'best.pt'}")
    print(f"[OK] Plots & curves: {save_dir}")


if __name__ == "__main__":
    main()
