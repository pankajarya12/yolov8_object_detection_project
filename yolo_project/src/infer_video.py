"""Run YOLOv8 inference on a video file with FPS overlay and CSV logging.

Example:
    python src/infer_video.py --weights runs/detect/train/weights/best.pt \\
        --source path/to/video.mp4 --conf 0.25
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from utils import (DetectionLogger, FPSMeter, count_per_class,
                   draw_detections, overlay_hud)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True, help="Path to video file")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--out", default="outputs/videos")
    p.add_argument("--device", default="")
    p.add_argument("--show", action="store_true", help="Display window while processing")
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    class_names = model.model.names if hasattr(model.model, "names") else model.names
    name_list = [class_names[i] for i in range(len(class_names))]

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"[ERR] Cannot open video: {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = out_dir / f"{src.stem}_pred.mp4"
    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    logger = DetectionLogger(out_dir / f"{src.stem}_detections.csv")
    meter = FPSMeter()
    frame_idx = 0

    print(f"[INFO] Processing {src.name} ({w}x{h} @ {fps:.1f}fps)")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        result = model.predict(frame, conf=args.conf, iou=args.iou,
                               device=args.device or None, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clses = result.boxes.cls.cpu().numpy().astype(int)

        annotated = draw_detections(frame, boxes, confs, clses, name_list)
        annotated = overlay_hud(annotated, meter.tick(), len(boxes))
        writer.write(annotated)
        logger.log(frame_idx, clses, name_list, confs, boxes)

        if args.show:
            cv2.imshow("YOLOv8 — Video", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_idx % 30 == 0:
            counts = count_per_class(clses, name_list)
            print(f"  frame {frame_idx} | objects={len(boxes)} | {counts}")

    cap.release()
    writer.release()
    logger.close()
    if args.show:
        cv2.destroyAllWindows()

    print(f"\n[OK] Annotated video: {out_path}")
    print(f"[OK] Detection log:   {out_dir / f'{src.stem}_detections.csv'}")


if __name__ == "__main__":
    main()
