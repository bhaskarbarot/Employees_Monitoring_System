#!/usr/bin/env python3
"""
mining.py — Hard-negative mining for Employee Monitor v4.

HardNegativeMiner:
  Background thread that samples frames from quiet (no-alert) camera periods.
  Runs the full detection pipeline on sampled frames.
  When the model fires a phone detection but no real phone exists (false positive),
  the frame is saved as a hard negative (class=working, label=the false bbox).

  This creates a targeted training signal: "here is exactly what confused the model;
  learn NOT to fire on these objects."

  Target objects commonly confused with phones from overhead:
    - Computer mice (rectangular, similar size)
    - ID badge lanyards
    - Coffee cups viewed from above
    - Sticky note pads
    - Keyboard sections

  Hard negatives are stored in training_data/hard_negatives/ and merged into
  training by train.py prepare_dataset().

  Rate limiting: max MAX_NEGATIVES_PER_HOUR per camera to avoid disk bloat.
"""

import json
import logging
import os
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_log = logging.getLogger("mining")

_BASE        = Path(__file__).parent.resolve()
NEG_IMG_DIR  = _BASE / "training_data" / "hard_negatives" / "images"
NEG_LBL_DIR  = _BASE / "training_data" / "hard_negatives" / "labels"
STATS_FILE   = _BASE / "training_data" / "hard_negatives" / "stats.json"

MINE_INTERVAL_SEC     = 60     # sample frames every 60 seconds
MAX_NEGATIVES_PER_HOUR = 20    # rate limit per camera
FRAMES_PER_SAMPLE     = 3      # frames to sample per camera per cycle
FALSE_POS_CONF_MIN    = 0.25   # minimum YOLO confidence to be considered a FP candidate
WORKING_CLASS_ID      = 4      # class 4 = "working" (negative sample in custom model)


class HardNegativeMiner(threading.Thread):
    """
    Background thread: samples frames when no phone alert is active and checks
    whether the model fires false positive phone detections.
    Confirmed FPs → saved as hard negatives with the offending bbox labeled as 'working'.
    """

    def __init__(self) -> None:
        super().__init__(daemon=True, name="HardNegMiner")
        # cam_id → list of (timestamp, frame_copy) tuples
        self._frame_buffer : Dict[int, List[Tuple[float, np.ndarray]]] = defaultdict(list)
        self._buffer_lock  = threading.Lock()
        # rate limiting: cam_id → list of timestamps of saved negatives this hour
        self._saved_times  : Dict[int, List[float]] = defaultdict(list)
        self._stats        : Dict[str, int]         = {"total": 0}
        self._stats_lock   = threading.Lock()
        NEG_IMG_DIR.mkdir(parents=True, exist_ok=True)
        NEG_LBL_DIR.mkdir(parents=True, exist_ok=True)

    def push_frame(self, cam_id: int, frame: np.ndarray,
                   has_active_alert: bool) -> None:
        """
        Call from camera reader (or BatchDetectWorker) when a frame is available.
        Frames are only buffered when NO phone alert is active — quiet periods only.
        """
        if has_active_alert:
            return
        with self._buffer_lock:
            buf = self._frame_buffer[cam_id]
            buf.append((time.time(), frame.copy()))
            # Keep only last 5 minutes of frames (300s at ~1fps sample = ~300 frames)
            cutoff = time.time() - 300
            self._frame_buffer[cam_id] = [(t, f) for t, f in buf if t > cutoff]

    def run(self) -> None:
        _log.info("[HardNegMiner] started — mining every %ds", MINE_INTERVAL_SEC)
        while True:
            time.sleep(MINE_INTERVAL_SEC)
            try:
                self._mine_cycle()
            except Exception as exc:
                _log.warning("[HardNegMiner] cycle error: %s", exc)

    def get_stats(self) -> Dict[str, int]:
        with self._stats_lock:
            return dict(self._stats)

    def _mine_cycle(self) -> None:
        try:
            from ultralytics import YOLO
            import torch
        except ImportError:
            return

        custom = _BASE / "custom_model" / "weights" / "best.pt"
        coco   = _BASE / "yolov8s.pt"
        if not custom.exists() or not coco.exists():
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"

        with self._buffer_lock:
            cam_buffers = {k: list(v) for k, v in self._frame_buffer.items()}

        for cam_id, buf in cam_buffers.items():
            if not buf:
                continue
            if not self._rate_ok(cam_id):
                continue

            # Sample FRAMES_PER_SAMPLE frames uniformly from buffer
            step = max(1, len(buf) // FRAMES_PER_SAMPLE)
            samples = [buf[i][1] for i in range(0, len(buf), step)][:FRAMES_PER_SAMPLE]

            for frame in samples:
                fp_bboxes = self._detect_false_positives(frame, custom, coco, device)
                if fp_bboxes:
                    self._save_negative(cam_id, frame, fp_bboxes)
                    if not self._rate_ok(cam_id):
                        break

    def _detect_false_positives(self, frame: np.ndarray,
                                custom_path: Path, coco_path: Path,
                                device: str) -> List[Tuple[float, float, float, float]]:
        """
        Returns list of (cx,cy,w,h) normalized bboxes where model detects
        a phone but confidence is in the ambiguous range (likely FP).
        """
        try:
            from ultralytics import YOLO
            fh, fw = frame.shape[:2]
            fps: List[Tuple] = []

            # Run COCO phone detector on frame
            det = YOLO(str(coco_path))
            r   = det([frame], verbose=False, conf=FALSE_POS_CONF_MIN,
                      imgsz=416, device=device, classes=[67])   # COCO phone

            if r and r[0].boxes:
                for box in r[0].boxes:
                    conf = float(box.conf[0])
                    # Ambiguous range: below CONF_PHONE (0.35) but above FP_CONF_MIN (0.25)
                    # These are exactly the detections we want to suppress with hard negatives
                    if FALSE_POS_CONF_MIN <= conf < 0.35:
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        cx = ((x1+x2)/2) / fw
                        cy = ((y1+y2)/2) / fh
                        bw = (x2-x1) / fw
                        bh = (y2-y1) / fh
                        fps.append((cx, cy, bw, bh))
            return fps[:3]  # cap at 3 FP bboxes per frame
        except Exception as exc:
            _log.debug("[HardNegMiner] detection error: %s", exc)
            return []

    def _save_negative(self, cam_id: int, frame: np.ndarray,
                       fp_bboxes: List[Tuple]) -> None:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:22]
        stem = f"neg_cam{cam_id}_{ts}"
        img_path = NEG_IMG_DIR / f"{stem}.jpg"
        lbl_path = NEG_LBL_DIR / f"{stem}.txt"

        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])

        # Label all FP bboxes as WORKING_CLASS_ID (class 4 = working/negative)
        lines = []
        for cx, cy, bw, bh in fp_bboxes:
            lines.append(f"{WORKING_CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        lbl_path.write_text("\n".join(lines) + "\n")

        now = time.time()
        self._saved_times[cam_id].append(now)

        with self._stats_lock:
            self._stats["total"] = self._stats.get("total", 0) + 1

        _log.info("[HardNegMiner] saved hard negative cam%d: %d FP bboxes → %s",
                  cam_id, len(fp_bboxes), stem)
        self._write_stats()

    def _rate_ok(self, cam_id: int) -> bool:
        """True if camera is below MAX_NEGATIVES_PER_HOUR rate."""
        cutoff = time.time() - 3600
        self._saved_times[cam_id] = [t for t in self._saved_times[cam_id] if t > cutoff]
        return len(self._saved_times[cam_id]) < MAX_NEGATIVES_PER_HOUR

    def _write_stats(self) -> None:
        with self._stats_lock:
            data = {"updated": datetime.now().isoformat(), "counts": dict(self._stats)}
        try:
            STATS_FILE.write_text(json.dumps(data, indent=2))
        except Exception:
            pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")
    miner = HardNegativeMiner()
    miner.start()
    print("HardNegativeMiner running.  Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(60)
            print("Stats:", miner.get_stats())
    except KeyboardInterrupt:
        pass
