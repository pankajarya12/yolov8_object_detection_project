"""Run YOLOv8 inference on a single image or folder of images.

Example:
    python src/infer_image.py --weights runs/detect/train/weights/best.pt \\
        --source path/to/img.jpg --conf 0.25
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from utils import DetectionLogger, count_per_class, draw_detections


IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True, help="Image file or folder")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--out", default="outputs/images")
    p.add_argument("--device", default="")
    return p.parse_args()


def gather_images(source: str):
    p = Path(source)
    if p.is_file():
        return [p]
    return sorted([f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT])


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    class_names = model.model.names if hasattr(model.model, "names") else model.names

    images = gather_images(args.source)
    if not images:
        raise SystemExit(f"[ERR] No images found under {args.source}")

    logger = DetectionLogger(out_dir / "detections.csv")

    for idx, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[WARN] Could not read {img_path}")
            continue

        result = model.predict(frame, conf=args.conf, iou=args.iou,
                               device=args.device or None, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clses = result.boxes.cls.cpu().numpy().astype(int)

        annotated = draw_detections(frame.copy(), boxes, confs, clses,
                                    [class_names[i] for i in range(len(class_names))])
        out_path = out_dir / f"{img_path.stem}_pred{img_path.suffix}"
        cv2.imwrite(str(out_path), annotated)

        logger.log(idx, clses, [class_names[i] for i in range(len(class_names))],
                   confs, boxes)

        counts = count_per_class(clses,
                                 [class_names[i] for i in range(len(class_names))])
        print(f"[{idx+1}/{len(images)}] {img_path.name} -> {out_path.name} | {counts}")

    logger.close()
    print(f"\n[OK] Annotated images saved to {out_dir}")
    print(f"[OK] Detection log: {out_dir / 'detections.csv'}")


if __name__ == "__main__":
    main()
