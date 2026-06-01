#!/usr/bin/env python3
"""
train.py — Fine-tune yolov8s on annotated training data.

Usage:
    python3 train.py           → train on all collected annotations
    python3 train.py --check   → show how many images are ready, then exit

Input:  training_data/images/*.jpg  +  training_data/labels/*.txt
Output: custom_model/weights/best.pt  (auto-loaded by monitor.py on restart)

Classes:
    0 = phone_hand   1 = phone_ear   2 = phone_desk
    3 = sleeping     4 = working
"""

import os, sys, shutil, random, yaml, argparse
from pathlib import Path
from datetime import datetime

TRAIN_DIR   = Path("training_data")
OUT_DIR     = Path("custom_model")
MIN_IMAGES  = 15   # won't train below this

CLASSES = ["phone_hand", "phone_ear", "phone_desk", "sleeping", "working"]
CLASS_IDS = {c: i for i, c in enumerate(CLASSES)}


def count_by_class():
    counts = {c: 0 for c in CLASSES}
    lbl_dir = TRAIN_DIR / "labels"
    if not lbl_dir.exists(): return counts
    for f in lbl_dir.glob("*.txt"):
        for line in f.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                idx = int(parts[0])
                if idx < len(CLASSES):
                    counts[CLASSES[idx]] += 1
    return counts


def total_images():
    img_dir = TRAIN_DIR / "images"
    return len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0


def prepare_dataset():
    """Split images 80/20 train/val and write dataset.yaml."""
    imgs = list((TRAIN_DIR / "images").glob("*.jpg"))
    random.shuffle(imgs)
    split = max(1, int(len(imgs) * 0.8))
    sets = {"train": imgs[:split], "val": imgs[split:]}

    for split_name, split_imgs in sets.items():
        img_out = TRAIN_DIR / "images" / split_name
        lbl_out = TRAIN_DIR / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img in split_imgs:
            shutil.copy2(img, img_out / img.name)
            lbl = TRAIN_DIR / "labels" / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, lbl_out / lbl.name)

    cfg = {
        "path":  str(TRAIN_DIR.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(CLASSES),
        "names": CLASSES,
    }
    yaml_path = TRAIN_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return len(sets["train"]), len(sets["val"]), yaml_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="Show stats only, no training")
    args = ap.parse_args()

    n      = total_images()
    counts = count_by_class()

    print(f"\n{'═'*55}")
    print(f"  Training Data Status")
    print(f"{'═'*55}")
    print(f"  Total images : {n}")
    for cls, cnt in counts.items():
        bar = "█" * min(cnt // 2, 20)
        print(f"  {cls:12}: {cnt:4}  {bar}")
    print(f"{'═'*55}")

    if args.check:
        if n < MIN_IMAGES:
            print(f"\n  Need at least {MIN_IMAGES} images to train.")
            print(f"  Use the Training UI to annotate more frames.\n")
        else:
            print(f"\n  Ready to train!  Run: python3 train.py\n")
        return

    if n < MIN_IMAGES:
        print(f"\n  Need at least {MIN_IMAGES} annotated images. Have {n}.")
        print(f"  Keep annotating in the Training UI.\n")
        sys.exit(1)

    print(f"\n  Starting training on {n} images...")
    print(f"  This will take 20-40 minutes on your GPU.\n")

    n_train, n_val, yaml_path = prepare_dataset()
    print(f"  Train: {n_train}  Val: {n_val}")

    from ultralytics import YOLO
    epochs = 50 if n < 100 else 100

    # Start from pre-trained weights for best accuracy
    base = "custom_model/weights/best.pt"
    start_weights = base if Path(base).exists() else "yolov8s.pt"
    print(f"  Base model   : {start_weights}")
    print(f"  Epochs       : {epochs}\n")

    model = YOLO(start_weights)
    model.train(
        data     = str(yaml_path),
        epochs   = epochs,
        imgsz    = 640,
        batch    = 8,
        lr0      = 0.0005,
        patience = 20,
        project  = str(OUT_DIR),
        name     = "weights",
        exist_ok = True,
        verbose  = True,
    )

    best = OUT_DIR / "weights" / "weights" / "best.pt"
    final = OUT_DIR / "weights" / "best.pt"

    # Flatten to custom_model/weights/best.pt
    if best.exists():
        shutil.copy2(best, final)

    if final.exists():
        print(f"\n{'═'*55}")
        print(f"  Training complete!")
        print(f"  Model saved → {final}")
        print(f"  Restart monitor.py to use the new model.")
        print(f"{'═'*55}\n")
    else:
        print("\n  Training finished but best.pt not found. Check errors above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
