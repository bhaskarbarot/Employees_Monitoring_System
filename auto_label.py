#!/usr/bin/env python3
"""
auto_label.py — Automated training data collection for Employee Monitor v4.

AutoLabelService:
  Background thread that scans training_data/verified/ every 30 seconds.
  For each OllamaVerifier-confirmed alert photo, runs the custom model at
  high resolution (imgsz=640). If model confidence > AUTO_CONF_THRESHOLD
  and the predicted class matches the filename event type → auto-copies
  the image and YOLO label to training_data/auto_labeled/.

  This creates a continuous data flywheel:
    Real alert → OllamaVerifier confirms → AutoLabelService labels → retraining

Usage:
    service = AutoLabelService()
    service.start()          # starts background thread
    service.get_stats()      # returns dict of counts
"""

import json
import logging
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

_log = logging.getLogger("auto_label")

_BASE        = Path(__file__).parent.resolve()
VERIFIED_DIR = _BASE / "training_data" / "verified"
AUTO_IMG_DIR = _BASE / "training_data" / "auto_labeled" / "images"
AUTO_LBL_DIR = _BASE / "training_data" / "auto_labeled" / "labels"
STATS_FILE   = _BASE / "training_data" / "auto_labeled" / "stats.json"

AUTO_CONF_THRESHOLD = 0.75   # minimum model confidence to accept auto-label
SCAN_INTERVAL_SEC   = 30     # how often to scan for new verified photos

# Map event name substring in filename → custom model class IDs
EVENT_TO_CLASS = {
    "PHONE_HAND": 0,
    "PHONE_EAR":  1,
    "SLEEPING":   3,
}
CLASS_NAMES = ["phone_hand", "phone_ear", "phone_desk", "sleeping", "working"]


class AutoLabelService(threading.Thread):
    """
    Background thread: converts OllamaVerifier-confirmed alert photos into
    YOLO-format training labels using the custom model as an oracle.
    """

    def __init__(self) -> None:
        super().__init__(daemon=True, name="AutoLabel")
        self._seen     : set          = set()
        self._stats    : Dict[str, int] = {c: 0 for c in CLASS_NAMES}
        self._stats_lock              = threading.Lock()
        AUTO_IMG_DIR.mkdir(parents=True, exist_ok=True)
        AUTO_LBL_DIR.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        _log.info("[AutoLabel] started — scanning every %ds", SCAN_INTERVAL_SEC)
        while True:
            try:
                self._scan()
            except Exception as exc:
                _log.warning("[AutoLabel] scan error: %s", exc)
            time.sleep(SCAN_INTERVAL_SEC)

    def get_stats(self) -> Dict[str, int]:
        with self._stats_lock:
            return dict(self._stats)

    def _scan(self) -> None:
        if not VERIFIED_DIR.exists():
            return

        # Lazy import to avoid loading heavy deps at module level
        try:
            from ultralytics import YOLO
            import torch
        except ImportError:
            return

        photos = sorted(VERIFIED_DIR.glob("Cam*.jpg"),
                        key=lambda p: p.stat().st_mtime, reverse=True)[:50]

        new_count = 0
        for photo in photos:
            if photo.name in self._seen:
                continue
            self._seen.add(photo.name)

            # Determine expected class from filename
            expected_class = self._event_class(photo.name)
            if expected_class is None:
                continue

            img = cv2.imread(str(photo))
            if img is None:
                continue

            result = self._infer(img, expected_class)
            if result is None:
                continue

            cls_id, conf, bbox_norm = result
            self._save(photo, img, cls_id, bbox_norm, conf)
            new_count += 1

        if new_count > 0:
            _log.info("[AutoLabel] labeled %d new photos", new_count)
            self._write_stats()

    @staticmethod
    def _event_class(filename: str) -> Optional[int]:
        for event, cls_id in EVENT_TO_CLASS.items():
            if event in filename:
                return cls_id
        return None

    @staticmethod
    def _infer(img: np.ndarray, expected_class: int) -> Optional[Tuple[int, float, Tuple]]:
        """Run custom model on image.  Returns (cls_id, conf, (cx,cy,w,h)) or None."""
        custom = _BASE / "custom_model" / "weights" / "best.pt"
        if not custom.exists():
            return None
        try:
            from ultralytics import YOLO
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model  = YOLO(str(custom))
            h, w   = img.shape[:2]
            results = model([img], verbose=False, conf=AUTO_CONF_THRESHOLD,
                            imgsz=640, device=device)
            if not results or results[0].boxes is None:
                return None

            # Find best matching detection of expected class
            best_conf, best_box, best_cls = 0.0, None, None
            for box in results[0].boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == expected_class and conf > best_conf:
                    best_conf = conf
                    best_cls  = cls
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    best_box = (cx, cy, bw, bh)

            if best_box is None:
                return None
            return (best_cls, best_conf, best_box)
        except Exception as exc:
            _log.debug("[AutoLabel] infer error: %s", exc)
            return None

    def _save(self, source_photo: Path, img: np.ndarray,
              cls_id: int, bbox_norm: Tuple, conf: float) -> None:
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:22]
        stem  = f"auto_{source_photo.stem}_{ts}"
        img_path = AUTO_IMG_DIR / f"{stem}.jpg"
        lbl_path = AUTO_LBL_DIR / f"{stem}.txt"

        cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cx, cy, bw, bh = bbox_norm
        lbl_path.write_text(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        _log.debug("[AutoLabel] saved %s  cls=%s conf=%.2f",
                   stem, CLASS_NAMES[cls_id], conf)
        with self._stats_lock:
            self._stats[CLASS_NAMES[cls_id]] = self._stats.get(CLASS_NAMES[cls_id], 0) + 1

    def _write_stats(self) -> None:
        with self._stats_lock:
            data = {"updated": datetime.now().isoformat(), "counts": dict(self._stats)}
        STATS_FILE.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")
    svc = AutoLabelService()
    svc.start()
    try:
        while True:
            time.sleep(30)
            print("AutoLabel stats:", svc.get_stats())
    except KeyboardInterrupt:
        pass
