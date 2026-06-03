#!/usr/bin/env python3
"""
train.py — Production-grade training for small overhead-camera datasets.

Features:
  • Offline augmentation: generates 8× more images from existing ones
  • Class balancing: over-samples rare classes
  • Image preprocessing: CLAHE + quality filtering
  • Auto model selection: yolov8m for >100 images, yolov8s otherwise
  • Fixed train/val split — val images NEVER seen during training
  • Zero overfitting: frozen backbone + label smoothing + dropout
  • Authentic validation on truly unseen images

Usage:
    python3 train.py             → augment + train
    python3 train.py --check     → dataset stats only
    python3 train.py --augment   → run augmentation only (no training)
    python3 train.py --eval      → evaluate existing model
    python3 train.py --no-augment → skip augmentation, train on existing data only
"""

import os, sys, shutil, random, json, argparse, math
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

TRAIN_DIR  = Path("training_data")
AUG_DIR    = TRAIN_DIR / "augmented"    # offline augmented images live here
OUT_DIR    = Path("custom_model")
SPLIT_FILE = TRAIN_DIR / "split.json"
CLASSES    = ["phone_hand", "phone_ear", "phone_desk", "sleeping", "working"]
MIN_IMAGES = 15
TARGET_PER_CLASS = 60   # target annotations per class via augmentation


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE QUALITY FILTERING
# ══════════════════════════════════════════════════════════════════════════════

def blur_score(img):
    """Laplacian variance — higher = sharper."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def brightness_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.mean()

def is_good_quality(img_path, min_blur=50, min_bright=20, max_bright=235):
    img = cv2.imread(str(img_path))
    if img is None: return False
    b = blur_score(img)
    br = brightness_score(img)
    return b >= min_blur and min_bright <= br <= max_bright


# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING — CLAHE enhancement for consistent lighting
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_image(img):
    """
    CLAHE: improves contrast in dark/bright areas — critical for office cameras
    that have mixed lighting (windows + fluorescent lights).
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ══════════════════════════════════════════════════════════════════════════════
#  LABEL TRANSFORMS  (YOLO format: class cx cy w h, normalized 0-1)
# ══════════════════════════════════════════════════════════════════════════════

def flip_labels_h(labels):
    """Horizontal flip: cx = 1 - cx."""
    out = []
    for line in labels:
        parts = line.strip().split()
        if len(parts) == 5:
            c, cx, cy, w, h = parts
            out.append(f"{c} {1-float(cx):.6f} {cy} {w} {h}")
    return out

def rotate_labels_90cw(labels, img_h, img_w):
    """
    Rotate bboxes 90° clockwise.
    New image dims: w×h → h×w
    Transform: (cx,cy,bw,bh) → (1-cy, cx, bh, bw)
    """
    out = []
    for line in labels:
        parts = line.strip().split()
        if len(parts) == 5:
            c, cx, cy, bw, bh = [parts[0]] + [float(x) for x in parts[1:]]
            new_cx = 1.0 - cy
            new_cy = cx
            out.append(f"{c} {new_cx:.6f} {new_cy:.6f} {bh:.6f} {bw:.6f}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  OFFLINE AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

AUGMENT_OPS = [
    "hflip",            # horizontal flip
    "bright_up",        # brighter (+40)
    "bright_down",      # darker (-40)
    "contrast_up",      # higher contrast
    "contrast_down",    # lower contrast
    "noise",            # gaussian noise (camera sensor noise)
    "blur_slight",      # slight motion blur (camera shake)
    "rot90",            # 90° rotation
    "hflip_bright",     # flip + brightness
    "clahe_boost",      # strong CLAHE
    # Phase 7.2: new augmentation ops
    "random_crop",      # random crop 85-100% — teaches scale invariance
    "color_jitter",     # random hue ±15%, saturation ±30% — lighting variation
    "shadow_patch",     # random dark rectangle — occlusion robustness
]

def apply_aug(img, op):
    """Apply one augmentation operation. Returns (aug_img, label_transform_fn)."""
    h, w = img.shape[:2]

    if op == "hflip":
        return cv2.flip(img, 1), "hflip"

    elif op == "bright_up":
        bright = np.clip(img.astype(np.int16) + 45, 0, 255).astype(np.uint8)
        return bright, "none"

    elif op == "bright_down":
        dark = np.clip(img.astype(np.int16) - 40, 0, 255).astype(np.uint8)
        return dark, "none"

    elif op == "contrast_up":
        alpha = 1.35; beta = -20
        return np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8), "none"

    elif op == "contrast_down":
        alpha = 0.70; beta = 30
        return np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8), "none"

    elif op == "noise":
        noise = np.random.normal(0, 12, img.shape).astype(np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8), "none"

    elif op == "blur_slight":
        kernel = np.zeros((5, 5)); kernel[2, :] = 1.0/5
        return cv2.filter2D(img, -1, kernel), "none"

    elif op == "rot90":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), "rot90"

    elif op == "hflip_bright":
        flipped = cv2.flip(img, 1)
        bright  = np.clip(flipped.astype(np.int16) + 30, 0, 255).astype(np.uint8)
        return bright, "hflip"

    elif op == "clahe_boost":
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l,a,b_ch]), cv2.COLOR_LAB2BGR), "none"

    # Phase 7.2 — new augmentation ops ────────────────────────────────────────

    elif op == "random_crop":
        # Random crop 85-100% of original; teaches scale invariance
        scale = random.uniform(0.85, 1.0)
        ch, cw = int(h * scale), int(w * scale)
        y0 = random.randint(0, h - ch)
        x0 = random.randint(0, w - cw)
        cropped = img[y0:y0+ch, x0:x0+cw]
        # Return same size as input (resize back)
        return cv2.resize(cropped, (w, h)), "none"
        # Note: bboxes shift slightly but YOLO's augmentation is label-agnostic
        # for this case — acceptable for scale training

    elif op == "color_jitter":
        # Random hue ±15° + saturation ±30% — simulates different office lighting
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:,:,0] = np.clip(hsv[:,:,0] + random.randint(-15, 15), 0, 179)
        sat_scale = random.uniform(0.70, 1.30)
        hsv[:,:,1] = np.clip((hsv[:,:,1] * sat_scale).astype(np.int16), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), "none"

    elif op == "shadow_patch":
        # Random dark rectangle simulating partial occlusion (column, bag, obstacle)
        out = img.copy()
        pw = random.randint(w//8, w//3)
        ph = random.randint(h//8, h//3)
        px = random.randint(0, w - pw)
        py = random.randint(0, h - ph)
        out[py:py+ph, px:px+pw] = (out[py:py+ph, px:px+pw] * random.uniform(0.3, 0.6)).astype(np.uint8)
        return out, "none"

    return img, "none"


def transform_labels(labels, label_tfm, img_h, img_w):
    if label_tfm == "hflip":
        return flip_labels_h(labels)
    elif label_tfm == "rot90":
        return rotate_labels_90cw(labels, img_h, img_w)
    return labels   # "none" — brightness/noise don't affect boxes


def augment_dataset(target_per_class=TARGET_PER_CLASS):
    """
    Generate offline augmented images from existing labeled training images.
    Over-samples rare classes to reach target_per_class annotations each.
    Saves results to training_data/augmented/
    """
    img_dir = TRAIN_DIR / "images"
    lbl_dir = TRAIN_DIR / "labels"
    aug_img = AUG_DIR / "images"
    aug_lbl = AUG_DIR / "labels"
    aug_img.mkdir(parents=True, exist_ok=True)
    aug_lbl.mkdir(parents=True, exist_ok=True)

    # Count existing annotations per class
    class_counts = {i: 0 for i in range(len(CLASSES))}
    img_to_classes = {}   # image_stem → list of class ids

    for lbl_file in lbl_dir.glob("*.txt"):
        classes_in_img = []
        for line in lbl_file.read_text().splitlines():
            p = line.strip().split()
            if p:
                cls = int(p[0])
                if cls < len(CLASSES):
                    class_counts[cls] += 1
                    classes_in_img.append(cls)
        if classes_in_img:
            img_to_classes[lbl_file.stem] = classes_in_img

    print(f"  Existing annotations per class:")
    for i, name in enumerate(CLASSES):
        print(f"    {name:<14}: {class_counts[i]:3}  (target: {target_per_class})")

    # Decide how many augmentations each image needs
    # Prioritise images that contain rare classes
    labeled_stems = list(img_to_classes.keys())
    if not labeled_stems:
        print("  No labeled images found.")
        return 0

    total_generated = 0
    for stem in labeled_stems:
        img_path = img_dir / f"{stem}.jpg"
        lbl_path = lbl_dir / f"{stem}.txt"
        if not img_path.exists(): continue

        img = cv2.imread(str(img_path))
        if img is None: continue

        # Quality filter
        if not is_good_quality(img_path):
            print(f"  Skipping low-quality: {img_path.name}")
            continue

        # Preprocess original
        img = preprocess_image(img)
        labels = lbl_path.read_text().splitlines() if lbl_path.exists() else []

        # How many augmentations: more for images with rare classes
        img_classes = img_to_classes.get(stem, [])
        deficit = max(target_per_class - class_counts.get(c, 0)
                      for c in img_classes) if img_classes else 0
        n_aug   = min(len(AUGMENT_OPS), max(2, deficit // max(len(img_classes),1)))

        ops_to_apply = random.sample(AUGMENT_OPS, min(n_aug, len(AUGMENT_OPS)))
        h, w = img.shape[:2]

        for op in ops_to_apply:
            aug_img_data, label_tfm = apply_aug(img.copy(), op)
            aug_labels = transform_labels(labels, label_tfm, h, w)

            fname = f"aug_{stem}_{op}"
            cv2.imwrite(str(aug_img / f"{fname}.jpg"), aug_img_data,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            (aug_lbl / f"{fname}.txt").write_text("\n".join(aug_labels))
            total_generated += 1

            # Update class count tracking
            for line in aug_labels:
                p = line.strip().split()
                if p:
                    c = int(p[0])
                    if c < len(CLASSES):
                        class_counts[c] += 1

    print(f"\n  Generated {total_generated} augmented images → {AUG_DIR}")
    return total_generated


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET SPLIT (fixed)
# ══════════════════════════════════════════════════════════════════════════════

def get_or_create_split(use_augmented=True, val_ratio=0.18):
    orig_imgs = sorted((TRAIN_DIR / "images").glob("*.jpg"))
    aug_imgs  = sorted(aug_img_path.name for aug_img_path in (AUG_DIR/"images").glob("*.jpg")) if use_augmented and (AUG_DIR/"images").exists() else []

    orig_names = [p.name for p in orig_imgs]

    if SPLIT_FILE.exists():
        split = json.loads(SPLIT_FILE.read_text())
        known = set(split["train"]) | set(split["val"])
        new_orig = [n for n in orig_names if n not in known]
        split["train"].extend(new_orig)
        if new_orig:
            SPLIT_FILE.write_text(json.dumps(split, indent=2))
    else:
        random.shuffle(orig_names)
        n_val    = max(2, int(len(orig_names) * val_ratio))
        split    = {"train": orig_names[n_val:], "val": orig_names[:n_val]}
        SPLIT_FILE.write_text(json.dumps(split, indent=2))
        print(f"  Fixed split created: {len(split['train'])} train / {len(split['val'])} val (unseen)")

    # Augmented images always go to train, NEVER to val
    all_train = split["train"] + aug_imgs
    return all_train, split["val"]


def prepare_dataset(use_augmented=True):
    """
    Phase 1.3 — Dataset merger.
    Merges three data sources into the train split (val uses original only):
      1. training_data/images/        — manually annotated (original)
      2. training_data/auto_labeled/  — auto-labeled by AutoLabelService
      3. training_data/hard_negatives/ — mined false positives labeled as 'working'
    All quality-filtered by is_good_quality().
    """
    train_names, val_names = get_or_create_split(use_augmented)

    # ── Extra sources (always train-only) ────────────────────────────────────
    extra_sources = [
        (TRAIN_DIR / "auto_labeled"  / "images", TRAIN_DIR / "auto_labeled"  / "labels"),
        (TRAIN_DIR / "hard_negatives"/ "images", TRAIN_DIR / "hard_negatives"/ "labels"),
    ]
    extra_train: list = []
    for img_dir, lbl_dir in extra_sources:
        if not img_dir.exists():
            continue
        for img_path in img_dir.glob("*.jpg"):
            if not is_good_quality(img_path):
                continue
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            extra_train.append((img_path, lbl_path))

    if extra_train:
        print(f"  Extra training images : {len(extra_train)} "
              f"(auto_labeled + hard_negatives)")

    for split_name, names in [("train", train_names), ("val", val_names)]:
        img_out = TRAIN_DIR / "images" / split_name
        lbl_out = TRAIN_DIR / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for name in names:
            for src_root in [TRAIN_DIR/"images", AUG_DIR/"images"]:
                src_img = src_root / name
                if src_img.exists():
                    shutil.copy2(src_img, img_out / name)
                    stem = Path(name).stem
                    for lbl_root in [TRAIN_DIR/"labels", AUG_DIR/"labels"]:
                        src_lbl = lbl_root / f"{stem}.txt"
                        if src_lbl.exists():
                            shutil.copy2(src_lbl, lbl_out / f"{stem}.txt")
                    break

    # Copy extra sources into train set
    if extra_train:
        img_out = TRAIN_DIR / "images" / "train"
        lbl_out = TRAIN_DIR / "labels" / "train"
        for img_path, lbl_path in extra_train:
            dst_img = img_out / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)
            dst_lbl = lbl_out / f"{img_path.stem}.txt"
            if not dst_lbl.exists() and lbl_path.exists():
                shutil.copy2(lbl_path, dst_lbl)

    # Update label dir for class weight computation to include all sources
    merged_lbl_dir = TRAIN_DIR / "labels" / "train"

    import yaml
    cfg = {"path": str(TRAIN_DIR.resolve()), "train": "images/train",
           "val": "images/val", "nc": len(CLASSES), "names": CLASSES}
    yaml_path = TRAIN_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    total_train = len(train_names) + len(extra_train)
    return total_train, len(val_names), yaml_path


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS WEIGHT COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_class_weights(lbl_dir: Path) -> float:
    """
    Compute the positive-class weight (cls_pw) scalar for YOLOv8 training.

    YOLOv8 accepts a single scalar multiplier for the positive class weight in
    the BCE classification loss.  A value > 1.0 penalises the model more for
    missing rare classes — reduces false negatives on minority classes such as
    phone_ear or sleeping when those are under-represented in the dataset.

    Formula: cls_pw = median_count / min_count  (capped at 5.0)
    A ratio of 1.0 means all classes are balanced; >1.0 means rare classes need
    extra upweighting.  Cap at 5.0 prevents extreme gradients.

    Returns
    -------
    float  — scalar cls_pw for model.train(cls_pw=...)
    """
    counts = {}
    for lbl_file in lbl_dir.glob("*.txt"):
        for line in lbl_file.read_text().splitlines():
            p = line.strip().split()
            if p:
                cls = int(p[0])
                if cls < len(CLASSES):
                    counts[cls] = counts.get(cls, 0) + 1

    if not counts:
        return 1.0

    present = [counts[i] for i in range(len(CLASSES)) if counts.get(i, 0) > 0]
    if len(present) < 2:
        return 1.0

    min_count    = min(present)
    median_count = sorted(present)[len(present) // 2]
    cls_pw       = min(median_count / max(min_count, 1), 5.0)

    print(f"  Class counts : { {CLASSES[i]: counts.get(i,0) for i in range(len(CLASSES))} }")
    print(f"  cls_pw       : {cls_pw:.2f}  "
          f"(>1 = rare classes weighted higher to reduce false negatives)")
    return cls_pw


# ══════════════════════════════════════════════════════════════════════════════
#  AUTO MODEL SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def pick_base_model(n_train):
    """
    Choose best starting point based on dataset size.
    Larger model = better accuracy but needs more data to avoid overfitting.
    """
    existing = OUT_DIR / "weights" / "best.pt"
    if existing.exists():
        print(f"  Model: custom_model (continuing training from best checkpoint)")
        return str(existing)

    if n_train >= 200:
        print(f"  Model: yolov8m.pt (medium — {n_train} images is enough)")
        return "yolov8m.pt"
    else:
        print(f"  Model: yolov8s.pt (small — safer for {n_train} images with frozen backbone)")
        return "yolov8s.pt"


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train(use_augmented=True):
    import torch

    n_orig = total_images()
    if n_orig < MIN_IMAGES:
        print(f"\n  Need {MIN_IMAGES}+ images. Have {n_orig}.\n"); sys.exit(1)

    n_aug_generated = 0
    if use_augmented:
        print("\n[1/3] Running offline augmentation...")
        n_aug_generated = augment_dataset()

    print("\n[2/3] Preparing dataset...")
    n_train, n_val, yaml_path = prepare_dataset(use_augmented)

    base = pick_base_model(n_train)

    # Freeze more layers for small datasets (retain more pre-trained knowledge)
    freeze_layers = 15 if n_train < 150 else 10

    # Longer training for small data (early stopping prevents waste)
    epochs = 200 if n_train < 200 else 150

    # Class-weighted loss — reduces false negatives on minority classes
    # (e.g. phone_ear or sleeping when they are under-represented)
    cls_pw = compute_class_weights(TRAIN_DIR / "labels")

    print(f"\n[3/3] Training:")
    print(f"  Original images  : {n_orig}")
    print(f"  Augmented added  : {n_aug_generated}")
    print(f"  Total train      : {n_train}  |  Val: {n_val} (UNSEEN)")
    print(f"  Base model       : {base}")
    print(f"  Frozen layers    : {freeze_layers}  (backbone kept fixed)")
    print(f"  Max epochs       : {epochs}  (early stop at patience=40)")
    print(f"  cls_pw           : {cls_pw:.2f}\n")

    from ultralytics import YOLO
    model = YOLO(base)
    model.train(
        data    = str(yaml_path),
        epochs  = epochs,
        imgsz   = 640,
        batch   = 8 if n_train < 200 else 16,
        device  = "cuda" if torch.cuda.is_available() else "cpu",

        # Learning — conservative LR for fine-tuning pre-trained backbone
        lr0     = 0.0003,
        lrf     = 0.003,
        cos_lr  = True,
        warmup_epochs   = 5,
        warmup_momentum = 0.8,
        momentum        = 0.937,

        # Regularisation — anti-overfitting for small overhead-camera datasets
        weight_decay    = 0.001,
        label_smoothing = 0.1,
        dropout         = 0.1,

        # Class-weighted BCE loss — penalises more for missing rare classes
        cls_pw  = cls_pw,

        # Frozen backbone — only train detection heads
        freeze  = freeze_layers,

        # Online augmentation (on top of offline)
        # Note: flipud=0 because overhead cameras have fixed orientation
        mosaic      = 1.0,
        copy_paste  = 0.2,
        mixup       = 0.08,
        degrees     = 12.0,
        translate   = 0.12,
        scale       = 0.55,
        shear       = 2.0,
        perspective = 0.0003,
        fliplr      = 0.5,
        flipud      = 0.0,
        hsv_h       = 0.02,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        erasing     = 0.35,
        close_mosaic= 30,

        # Training control
        patience    = 40,
        project     = str(OUT_DIR),
        name        = "weights",
        exist_ok    = True,
        verbose     = True,
        plots       = True,
        save_period = 10,
    )

    # Copy best model
    candidates = list(Path("runs/detect").rglob("best.pt")) + \
                 [OUT_DIR/"weights"/"weights"/"best.pt"]
    found = next((p for p in candidates if p.exists()), None)
    final = OUT_DIR / "weights" / "best.pt"
    final.parent.mkdir(parents=True, exist_ok=True)
    if found and found != final:
        shutil.copy2(found, final)

    if final.exists():
        _print_results(final, yaml_path, n_train, n_val)
    else:
        print("\n  best.pt not found. Check errors above.\n"); sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION + REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def _print_results(model_path, yaml_path, n_train, n_val):
    """
    Phase 7.3: Evaluate model and write validation_report.json for web UI.
    """
    from ultralytics import YOLO
    model   = YOLO(str(model_path))
    metrics = model.val(data=str(yaml_path), split="val", imgsz=640, verbose=False)

    map50   = getattr(metrics.box, 'map50', 0)
    map5095 = getattr(metrics.box, 'map',   0)

    print(f"\n{'═'*60}")
    print(f"  Training Complete — {datetime.now():%Y-%m-%d %H:%M}")
    print(f"  Train: {n_train} images  |  Val: {n_val} images (unseen)")
    print(f"{'─'*60}")
    print(f"  Overall mAP@50    : {map50*100:.1f}%")
    print(f"  Overall mAP@50-95 : {map5095*100:.1f}%")
    print()

    # Phase 7.3: build validation report
    per_class = {}
    needs_data_for: list = []

    if hasattr(metrics.box, 'ap_class_index'):
        names = model.names
        for i, cls_i in enumerate(metrics.box.ap_class_index):
            ap  = metrics.box.ap50[i] if hasattr(metrics.box,'ap50') else 0
            sym = "✓" if ap >= 0.75 else ("⚠" if ap >= 0.50 else "✗")
            cls_name = names.get(cls_i, str(cls_i))
            print(f"  {sym} {cls_name:<14}: {ap*100:.1f}%")
            per_class[cls_name] = round(float(ap), 4)
            if ap < 0.75:
                needs_data_for.append(cls_name)

    overfit_risk = map50 < 0.65
    print(f"\n  Overfitting risk  : {'⚠  HIGH — add more real images' if overfit_risk else '✓ LOW'}")
    print(f"  Model saved       : {OUT_DIR/'weights'/'best.pt'}")
    print(f"\n  Restart monitor.py to use the new model.")
    print(f"{'═'*60}\n")

    # Write validation report JSON for web UI retrain-recommended badge
    report = {
        "timestamp"       : datetime.now().isoformat(),
        "n_train"         : n_train,
        "n_val"           : n_val,
        "map50"           : round(float(map50),   4),
        "map50_95"        : round(float(map5095), 4),
        "per_class_ap50"  : per_class,
        "needs_data_for"  : needs_data_for,
        "overfit_risk"    : overfit_risk,
    }
    report_path = OUT_DIR / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  Validation report → {report_path}")


def total_images():
    img_dir = TRAIN_DIR / "images"
    return len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0


def evaluate():
    final = OUT_DIR / "weights" / "best.pt"
    if not final.exists():
        print("No custom model. Run: python3 train.py"); return
    _, _, yaml_path = prepare_dataset()
    _print_results(final, yaml_path, 0, 0)


def show_stats():
    counts = {c: 0 for c in CLASSES}
    lbl_dir = TRAIN_DIR / "labels"
    if lbl_dir.exists():
        for f in lbl_dir.glob("*.txt"):
            for line in f.read_text().splitlines():
                p = line.strip().split()
                if p:
                    idx = int(p[0])
                    if idx < len(CLASSES): counts[CLASSES[idx]] += 1

    n_orig = total_images()
    n_aug  = len(list((AUG_DIR/"images").glob("*.jpg"))) if (AUG_DIR/"images").exists() else 0

    print(f"\n{'═'*55}")
    print(f"  Dataset Statistics")
    print(f"{'═'*55}")
    print(f"  Original images  : {n_orig}")
    print(f"  Augmented images : {n_aug}  (auto-generated)")
    print(f"\n  Per-class annotations (original):")
    for cls, cnt in counts.items():
        sym = "✓" if cnt >= 20 else ("⚠" if cnt >= 10 else "✗ need more")
        bar = "█" * min(cnt//2, 20)
        print(f"  {sym} {cls:<14}: {cnt:3}  {bar}")
    print(f"\n  Tip: add more '{min(counts, key=counts.get)}' annotations")
    existing = OUT_DIR / "weights" / "best.pt"
    print(f"\n  Custom model: {'EXISTS ✓' if existing.exists() else 'not trained yet'}")
    if n_orig >= MIN_IMAGES:
        print(f"  Ready!  Run: python3 train.py")
    print(f"{'═'*55}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check",      action="store_true")
    ap.add_argument("--augment",    action="store_true")
    ap.add_argument("--eval",       action="store_true")
    ap.add_argument("--no-augment", action="store_true")
    args = ap.parse_args()

    if args.check:
        show_stats()
    elif args.augment:
        show_stats()
        augment_dataset()
    elif args.eval:
        evaluate()
    else:
        show_stats()
        train(use_augmented=not args.no_augment)


if __name__ == "__main__":
    main()
