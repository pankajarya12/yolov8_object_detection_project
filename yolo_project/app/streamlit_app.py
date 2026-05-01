"""Streamlit web demo for YOLOv8 detection.

Run:
    streamlit run app/streamlit_app.py
"""
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# allow `from utils import ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils import FPSMeter, count_per_class, draw_detections, overlay_hud  # noqa: E402


st.set_page_config(page_title="YOLOv8 Object Detection", page_icon="🎯", layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Settings")
default_weights = "yolov8n.pt"
weights_path = st.sidebar.text_input("Model weights (.pt)", value=default_weights)
conf = st.sidebar.slider("Confidence", 0.05, 0.95, 0.25, 0.05)
iou = st.sidebar.slider("IoU (NMS)", 0.1, 0.9, 0.45, 0.05)
mode = st.sidebar.radio("Source", ["Image", "Video", "Webcam"], horizontal=False)

@st.cache_resource(show_spinner="Loading YOLOv8 model…")
def load_model(path: str):
    return YOLO(path)


def run_detection(model: YOLO, frame_bgr: np.ndarray):
    result = model.predict(frame_bgr, conf=conf, iou=iou, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clses = result.boxes.cls.cpu().numpy().astype(int)
    names_dict = model.model.names if hasattr(model.model, "names") else model.names
    names = [names_dict[i] for i in range(len(names_dict))]
    annotated = draw_detections(frame_bgr.copy(), boxes, confs, clses, names)
    return annotated, boxes, confs, clses, names


# ---------------- Header ----------------
st.title("🎯 Real-Time Object Detection — YOLOv8")
st.caption("Upload an image / video, or use your webcam. Adjust confidence in the sidebar.")

if not Path(weights_path).exists() and weights_path == default_weights:
    st.info("First run will auto-download `yolov8n.pt` (~6 MB).")

try:
    model = load_model(weights_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ---------------- Image mode ----------------
if mode == "Image":
    file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if file is not None:
        img = Image.open(file).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        annotated, boxes, confs, clses, names = run_detection(model, frame_bgr)

        col1, col2 = st.columns(2)
        col1.image(img, caption="Original", use_container_width=True)
        col2.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                   caption=f"Detections: {len(boxes)}", use_container_width=True)

        counts = count_per_class(clses, names)
        if counts:
            st.subheader("📊 Class counts")
            st.bar_chart(pd.DataFrame({"count": counts}))
            df = pd.DataFrame({
                "class": [names[c] for c in clses],
                "confidence": [round(float(c), 3) for c in confs],
                "x1": boxes[:, 0].astype(int), "y1": boxes[:, 1].astype(int),
                "x2": boxes[:, 2].astype(int), "y2": boxes[:, 3].astype(int),
            })
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No objects detected — try lowering confidence.")

        ok, buf = cv2.imencode(".jpg", annotated)
        if ok:
            st.download_button("⬇️ Download annotated image",
                               data=buf.tobytes(),
                               file_name="annotated.jpg", mime="image/jpeg")

# ---------------- Video mode ----------------
elif mode == "Video":
    file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
    if file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(file.read())
        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error("Cannot open video.")
            st.stop()

        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_in, (w, h))

        progress = st.progress(0, text="Processing video…")
        slot = st.empty()
        meter = FPSMeter()
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            annotated, boxes, _, _, _ = run_detection(model, frame)
            annotated = overlay_hud(annotated, meter.tick(), len(boxes))
            writer.write(annotated)
            if idx % 5 == 0:
                slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                           channels="RGB", use_container_width=True)
                progress.progress(min(idx / total, 1.0),
                                  text=f"Processing… {idx}/{total}")
        cap.release()
        writer.release()
        progress.empty()
        st.success("✅ Done")
        with open(out_path, "rb") as f:
            st.download_button("⬇️ Download annotated video",
                               data=f.read(),
                               file_name="annotated.mp4", mime="video/mp4")

# ---------------- Webcam mode ----------------
else:
    st.info("Click **Start** to capture from your webcam (browser will ask permission). "
            "Use 'Stop' to end the stream.")
    snap = st.camera_input("Take a snapshot")
    if snap is not None:
        img = Image.open(snap).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        annotated, boxes, confs, clses, names = run_detection(model, frame_bgr)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                 caption=f"Detections: {len(boxes)}", use_container_width=True)
        if len(boxes):
            st.json(count_per_class(clses, names))

st.sidebar.markdown("---")
st.sidebar.caption("Built with Ultralytics YOLOv8 · OpenCV · Streamlit")
