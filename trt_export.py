#!/usr/bin/env python3
"""
trt_export.py — TensorRT model export for Employee Monitor v4.

Exports the three inference models to TensorRT FP16 engines for 2-4× speedup.
Engine files are placed alongside the source .pt files.

Usage:
    python3 trt_export.py                  # export all models
    python3 trt_export.py --detect-only    # only yolov8s.pt
    python3 trt_export.py --pose-only      # only yolov8n-pose.pt
    python3 trt_export.py --custom-only    # only custom_model/weights/best.pt
    python3 trt_export.py --check          # check which engines exist

Engine files created:
    yolov8s.engine                          (detect model, batch=4, imgsz=416)
    yolov8n-pose.engine                     (pose model,   batch=4, imgsz=416)
    custom_model/weights/best.engine        (custom model, batch=8, imgsz=320)

monitor.py auto-detects these files: if they exist AND tensorrt is importable,
the engine is loaded instead of the .pt model.  Falls back to .pt on any error.

Requirements:
    - NVIDIA GPU with TensorRT installed (comes with CUDA Toolkit / Jetson)
    - ultralytics >= 8.0 with tensorrt export support
    - pip install tensorrt  (or installed via CUDA toolkit)
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [trt_export] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
_log = logging.getLogger("trt_export")

_BASE = Path(__file__).parent.resolve()

MODELS = {
    "detect": {
        "src" : _BASE / "yolov8s.pt",
        "dst" : _BASE / "yolov8s.engine",
        "imgsz": 416,
        "batch": 4,
        "desc": "COCO person+phone detector",
    },
    "pose": {
        "src" : _BASE / "yolov8n-pose.pt",
        "dst" : _BASE / "yolov8n-pose.engine",
        "imgsz": 416,
        "batch": 4,
        "desc": "YOLOv8 nano pose estimator",
    },
    "custom": {
        "src" : _BASE / "custom_model" / "weights" / "best.pt",
        "dst" : _BASE / "custom_model" / "weights" / "best.engine",
        "imgsz": 320,
        "batch": 8,
        "desc": "Custom phone/sleep classifier (person crops)",
    },
}


def check_engines() -> None:
    print("\n  TensorRT engine status:")
    for name, cfg in MODELS.items():
        src_ok = cfg["src"].exists()
        dst_ok = cfg["dst"].exists()
        age_str = ""
        if dst_ok:
            age_h = (time.time() - cfg["dst"].stat().st_mtime) / 3600
            age_str = f"  ({age_h:.1f}h old)"
        print(f"    {name:<8}: src={'✓' if src_ok else '✗'}  "
              f"engine={'✓' + age_str if dst_ok else '✗ NOT EXPORTED'}")

    try:
        import tensorrt
        print(f"\n  TensorRT version: {tensorrt.__version__}  ✓")
    except ImportError:
        print("\n  TensorRT: NOT INSTALLED  ✗")
        print("  Install: pip install tensorrt  or use CUDA Toolkit installer")
    print()


def export_model(name: str, cfg: dict) -> bool:
    """Export one model to TensorRT engine.  Returns True on success."""
    src  = cfg["src"]
    dst  = cfg["dst"]
    if not src.exists():
        _log.warning("[%s] source model not found: %s", name, src)
        return False

    _log.info("[%s] Exporting %s → %s", name, src.name, dst.name)
    _log.info("[%s]   batch=%d  imgsz=%d  fp16=True", name, cfg["batch"], cfg["imgsz"])

    t0 = time.time()
    try:
        from ultralytics import YOLO
        model  = YOLO(str(src))
        result = model.export(
            format  = "engine",
            imgsz   = cfg["imgsz"],
            half    = True,          # FP16
            batch   = cfg["batch"],
            workspace = 4,           # GiB workspace for TRT optimizer
            verbose = False,
        )
        elapsed = time.time() - t0
        _log.info("[%s] Export done in %.0fs → %s", name, elapsed, dst)
        return True
    except Exception as exc:
        _log.error("[%s] Export FAILED: %s", name, exc)
        _log.info("[%s] Fallback: monitor.py will use .pt model", name)
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Export YOLOv8 models to TensorRT engines")
    ap.add_argument("--detect-only", action="store_true")
    ap.add_argument("--pose-only",   action="store_true")
    ap.add_argument("--custom-only", action="store_true")
    ap.add_argument("--check",       action="store_true", help="Show engine status only")
    args = ap.parse_args()

    if args.check:
        check_engines(); return

    to_export = list(MODELS.keys())
    if args.detect_only: to_export = ["detect"]
    if args.pose_only:   to_export = ["pose"]
    if args.custom_only: to_export = ["custom"]

    # Verify TensorRT available before starting
    try:
        import tensorrt
        _log.info("TensorRT %s available", tensorrt.__version__)
    except ImportError:
        _log.error("TensorRT not installed.  Run: pip install tensorrt")
        _log.info("Continuing — ultralytics may still export using its bundled TRT")

    results = {}
    for name in to_export:
        results[name] = export_model(name, MODELS[name])

    print("\n  Export summary:")
    for name, ok in results.items():
        status = "✓ SUCCESS" if ok else "✗ FAILED (will use .pt fallback)"
        print(f"    {name:<8}: {status}")
    print()

    if any(results.values()):
        print("  Restart monitor.py to use TensorRT engines.")
        print("  monitor.py auto-detects .engine files at startup.")
    print()


if __name__ == "__main__":
    main()
