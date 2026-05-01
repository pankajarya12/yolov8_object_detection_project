"""Live webcam inference with YOLOv8 — press 'q' to quit, 's' to save snapshot.

Example:
    python src/infer_webcam.py --weights yolov8n.pt --camera 0
"""
import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from utils import FPSMeter, count_per_class, draw_detections, overlay_hud


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--out", default="outputs/webcam")
    p.add_argument("--device", default="")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    class_names = model.model.names if hasattr(model.model, "names") else model.names
    name_list = [class_names[i] for i in range(len(class_names))]

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"[ERR] Cannot open camera index {args.camera}")

    meter = FPSMeter()
    print("[INFO] Press 'q' to quit, 's' to save snapshot.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Frame grab failed.")
            break

        result = model.predict(frame, conf=args.conf, iou=args.iou,
                               device=args.device or None, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clses = result.boxes.cls.cpu().numpy().astype(int)

        annotated = draw_detections(frame, boxes, confs, clses, name_list)
        annotated = overlay_hud(annotated, meter.tick(), len(boxes))
        cv2.imshow("YOLOv8 — Webcam (q=quit, s=save)", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            ts = time.strftime("%Y%m%d_%H%M%S")
            snap = out_dir / f"snap_{ts}.jpg"
            cv2.imwrite(str(snap), annotated)
            print(f"[OK] Saved snapshot: {snap} | {count_per_class(clses, name_list)}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
