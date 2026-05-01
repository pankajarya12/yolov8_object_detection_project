"""Split a flat folder of images+labels into train/val (default 80/20)."""
import argparse
import random
import shutil
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="Folder with all images")
    p.add_argument("--labels", required=True, help="Folder with all YOLO .txt labels")
    p.add_argument("--out", required=True, help="Output dataset root")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    images = [p for p in Path(args.images).iterdir() if p.suffix.lower() in IMG_EXT]
    random.shuffle(images)
    n_val = int(len(images) * args.val_ratio)
    splits = {"val": images[:n_val], "train": images[n_val:]}

    out = Path(args.out)
    for split, files in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in files:
            shutil.copy2(img, out / "images" / split / img.name)
            lbl = Path(args.labels) / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, out / "labels" / split / lbl.name)
        print(f"[OK] {split}: {len(files)} files")
    print(f"\n[OK] Dataset prepared at {out}")


if __name__ == "__main__":
    main()
