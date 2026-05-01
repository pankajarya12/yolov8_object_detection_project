"""Evaluate a trained YOLOv8 model and print mAP, Precision, Recall.

Example:
    python src/evaluate.py --weights runs/detect/train/weights/best.pt \\
        --data configs/data.yaml
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate YOLOv8 model")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--out", type=str, default="outputs/eval")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        plots=True,
        save_json=True,
    )

    summary = {
        "mAP50":      float(metrics.box.map50),
        "mAP50_95":   float(metrics.box.map),
        "precision":  float(metrics.box.mp),
        "recall":     float(metrics.box.mr),
        "fitness":    float(metrics.fitness),
    }

    print("\n=========== EVALUATION ============")
    for k, v in summary.items():
        print(f"  {k:12s}: {v:.4f}")
    print("===================================\n")

    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"[OK] Metrics saved to {out_dir / 'metrics.json'}")
    print(f"[OK] Plots saved to   {metrics.save_dir}")


if __name__ == "__main__":
    main()
