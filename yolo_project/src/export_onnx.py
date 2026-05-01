"""Export a trained YOLOv8 model to ONNX for cross-platform deployment.

Example:
    python src/export_onnx.py --weights runs/detect/train/weights/best.pt
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=12)
    p.add_argument("--dynamic", action="store_true")
    p.add_argument("--simplify", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    out = model.export(format="onnx", imgsz=args.imgsz,
                       opset=args.opset, dynamic=args.dynamic,
                       simplify=args.simplify)
    print(f"[OK] Exported ONNX model: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
