"""Dataset sanity checker for YOLO-format datasets.

Verifies:
  * data.yaml parses and required keys exist
  * images & labels exist and pair up
  * label rows are valid (class_id in range, coords in 0..1)
  * detects corrupt/unreadable images

Example:
    python src/dataset_check.py --data configs/data.yaml
"""
import argparse
from pathlib import Path

import cv2
import yaml


IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to data.yaml")
    return p.parse_args()


def check_split(split_name: str, img_dir: Path, label_dir: Path, nc: int):
    print(f"\n[ {split_name.upper()} ]  images={img_dir}  labels={label_dir}")
    if not img_dir.exists():
        print(f"  [ERR] Image dir missing: {img_dir}")
        return
    if not label_dir.exists():
        print(f"  [WARN] Label dir missing: {label_dir}")

    images = [f for f in img_dir.rglob("*") if f.suffix.lower() in IMG_EXT]
    issues = {"corrupt": 0, "missing_label": 0, "bad_label": 0, "ok": 0}

    for img in images:
        if cv2.imread(str(img)) is None:
            issues["corrupt"] += 1
            print(f"  [CORRUPT] {img.name}")
            continue
        lbl = label_dir / f"{img.stem}.txt"
        if not lbl.exists():
            issues["missing_label"] += 1
            continue
        bad = False
        for line_no, line in enumerate(lbl.read_text().splitlines(), 1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                print(f"  [BAD] {lbl.name}:{line_no} expected 5 values, got {len(parts)}")
                bad = True
                break
            try:
                cid = int(parts[0])
                vals = [float(x) for x in parts[1:]]
            except ValueError:
                print(f"  [BAD] {lbl.name}:{line_no} non-numeric values")
                bad = True
                break
            if cid < 0 or cid >= nc:
                print(f"  [BAD] {lbl.name}:{line_no} class_id {cid} not in [0, {nc - 1}]")
                bad = True
                break
            if not all(0.0 <= v <= 1.0 for v in vals):
                print(f"  [BAD] {lbl.name}:{line_no} coords out of [0,1]: {vals}")
                bad = True
                break
        if bad:
            issues["bad_label"] += 1
        else:
            issues["ok"] += 1

    print(f"  -> total={len(images)}  ok={issues['ok']}  "
          f"corrupt={issues['corrupt']}  missing_label={issues['missing_label']}  "
          f"bad_label={issues['bad_label']}")


def main():
    args = parse_args()
    cfg_path = Path(args.data).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())

    base = Path(cfg.get("path", cfg_path.parent)).expanduser()
    if not base.is_absolute():
        base = (cfg_path.parent / base).resolve()

    nc = int(cfg.get("nc", 0))
    names = cfg.get("names", {})
    print(f"[INFO] data.yaml: {cfg_path}")
    print(f"[INFO] nc={nc}, names={len(names)} classes")
    if nc != len(names):
        print(f"[WARN] nc ({nc}) does not match number of names ({len(names)})")

    for split in ("train", "val"):
        rel = cfg.get(split)
        if not rel:
            print(f"[WARN] '{split}' not defined in YAML")
            continue
        img_dir = (base / rel).resolve()
        label_dir = Path(str(img_dir).replace("images", "labels", 1))
        check_split(split, img_dir, label_dir, nc)

    print("\n[OK] Dataset check complete.")


if __name__ == "__main__":
    main()
