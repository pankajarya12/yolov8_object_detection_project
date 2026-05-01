#!/usr/bin/env bash
# Download the COCO128 mini dataset for quick experiments.
set -e
mkdir -p data
cd data
if [ ! -d "coco128" ]; then
  echo "[INFO] Downloading COCO128…"
  curl -L -o coco128.zip https://ultralytics.com/assets/coco128.zip
  unzip -q coco128.zip
  rm coco128.zip
  echo "[OK] Dataset ready at data/coco128"
else
  echo "[INFO] data/coco128 already exists. Skipping."
fi
