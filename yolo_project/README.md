# 🚗 Real-Time Object Detection using YOLOv8

End-to-end Object Detection system built with **Ultralytics YOLOv8**, **PyTorch**, and **OpenCV**.
Supports training on custom datasets, evaluation with mAP/Precision/Recall, and inference on
**images, videos, and live webcam** — plus a **Streamlit web demo**.

> Final Project — DS / AIML Capstone (Batch B65)

---

## 📌 Features

### Core (per project spec)
- ✅ Train custom YOLOv8 detector (n / s / m / l / x)
- ✅ Evaluate with **mAP@0.5**, **mAP@0.5:0.95**, Precision, Recall
- ✅ Inference on **images, videos, webcam** with annotated outputs
- ✅ Clean YOLO-format dataset preparation pipeline
- ✅ Configurable hyperparameters (epochs, imgsz, batch, model)

### 🌟 Extra Features (bonus)
- 🎨 **Streamlit Web App** — drag-and-drop image/video, live webcam, adjustable confidence
- 📊 **Auto-generated evaluation report** (PNG charts: PR curve, confusion matrix, loss curves)
- 🎥 **Real-time FPS counter** overlaid on video / webcam output
- 🔢 **Object counter & class-wise statistics** per frame
- 🧮 **IoU + NMS visualizer** (educational utility)
- 📦 **ONNX export** script for cross-platform deployment
- 🧹 **Dataset sanity checker** (corrupt images, missing labels, invalid class IDs)
- 📝 **Inference logs** as CSV (timestamp, class, confidence, bbox)
- 🎯 **Region-of-Interest (ROI) filtering** — count only objects inside a defined zone
- 🧪 Unit tests for utility functions

---

## 🗂️ Project Structure

```
yolo_project/
├── src/                  # Core source code
│   ├── train.py          # Training script
│   ├── evaluate.py       # Validation + metrics
│   ├── infer_image.py    # Image inference
│   ├── infer_video.py    # Video inference
│   ├── infer_webcam.py   # Live webcam inference
│   ├── export_onnx.py    # ONNX export
│   ├── utils.py          # Helpers (IoU, NMS, drawing, FPS)
│   └── dataset_check.py  # Dataset sanity checker
├── app/
│   └── streamlit_app.py  # Web demo (image/video/webcam)
├── configs/
│   └── data.yaml         # Dataset YAML (paths + class names)
├── notebooks/
│   └── 01_training_demo.ipynb
├── scripts/
│   ├── download_coco128.sh
│   └── prepare_dataset.py
├── data/                 # Datasets (gitignored)
├── outputs/              # Annotated images/videos + logs
├── runs/                 # Training runs (auto by Ultralytics)
├── tests/
│   └── test_utils.py
├── docs/
│   └── EVALUATION_REPORT.md
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Clone & enter project
git clone <your-repo-url>
cd yolo_project

# 2. Create virtual env (Python 3.9+)
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

Verify install:
```bash
python -c "import ultralytics; ultralytics.checks()"
```

---

## 📥 Dataset

### Option A — COCO128 (recommended for quick start)
```bash
bash scripts/download_coco128.sh
```

### Option B — Custom Dataset
1. Annotate using **LabelImg / Roboflow / CVAT** in YOLO format.
2. Folder layout:
   ```
   data/custom/
   ├── images/train/
   ├── images/val/
   ├── labels/train/
   └── labels/val/
   ```
3. Update `configs/data.yaml` with your class names.
4. Sanity check:
   ```bash
   python src/dataset_check.py --data configs/data.yaml
   ```

---

## 🏋️ Training

```bash
python src/train.py --data configs/data.yaml --model yolov8n.pt --epochs 50 --imgsz 640 --batch 16
```

Outputs land in `runs/detect/train*/` — including `weights/best.pt`.

---

## 📊 Evaluation

```bash
python src/evaluate.py --weights runs/detect/train/weights/best.pt --data configs/data.yaml
```

Generates a metrics table and saves charts to `outputs/eval/`.

---

## 🔍 Inference

**Image**
```bash
python src/infer_image.py --weights runs/detect/train/weights/best.pt --source path/to/img.jpg
```

**Video**
```bash
python src/infer_video.py --weights runs/detect/train/weights/best.pt --source path/to/video.mp4
```

**Webcam (live)**
```bash
python src/infer_webcam.py --weights runs/detect/train/weights/best.pt --camera 0
```

All annotated outputs save to `outputs/` along with a CSV log.

---

## 🌐 Streamlit Web Demo

```bash
streamlit run app/streamlit_app.py
```

Features: drag-and-drop image/video, live webcam, confidence slider, class filter,
download annotated output.

---

## 📦 Export to ONNX

```bash
python src/export_onnx.py --weights runs/detect/train/weights/best.pt
```

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 📈 Results (sample)

| Metric        | Value  |
|---------------|--------|
| mAP@0.5       | 0.842  |
| mAP@0.5:0.95  | 0.612  |
| Precision     | 0.87   |
| Recall        | 0.81   |
| FPS (CPU)     | ~12    |
| FPS (GPU)     | ~75    |

> Replace with your actual numbers after training. See `docs/EVALUATION_REPORT.md`.

---

## 🛠️ Tech Stack

`Python 3.9+` · `Ultralytics YOLOv8` · `PyTorch` · `OpenCV` · `Streamlit` · `NumPy` · `Pandas` · `Matplotlib`

---

## 📝 License

MIT — free to use for learning & non-commercial projects.

## 🙋 Author

Final Project Submission — GUVI / HCL Data Science Capstone (B65)
