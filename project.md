# Employee Monitor — Complete Project Blueprint (v3)

> **Last updated: 2026-06-02**  
> This document is a full technical specification of the project as it exists right now — every file, class, function, configuration value, hyperparameter, algorithm, bug fix, and design decision is documented here. An LLM given only this file should be able to recreate the project exactly as it currently stands.

---

## 1. Project Overview

**Name:** Employee Monitor v3 — Multi-Camera, GPU-Efficient, Production-Grade  
**Purpose:** Real-time AI-powered CCTV employee behaviour monitoring via RTSP streams. Detects phone usage (hand and ear), sleeping employees, and generates annotated photo evidence with duration tracking.  
**Language:** Python 3.8+ (tested on 3.10.12)  
**Primary entry point:** `monitor.py`  
**Web dashboard:** `web_ui.py` (Flask + gunicorn, port 5000)  
**Camera type:** Dahua NVR (multi-channel), Hikvision also supported  
**Hardware:** NVIDIA GPU required for full speed; CPU fallback available  
**Operating mode:** Always headless (no cv2 window); web browser is the only UI

### Version History

| Version | Changes |
|---------|---------|
| v1 | Original single-camera, basic YOLOv8 phone detection |
| v2 | Multi-camera batch inference, custom model, session timing, Ollama verifier |
| v3 (current) | KalmanTracker, CLAHE+EMA motion, absolute paths, stale frame detection, phone-on-ear fix, structured logging, class-weighted training |

---

## 2. Directory Structure

```
employees/
├── monitor.py               # Main AI detection engine — v3
├── web_ui.py                # Flask web dashboard + annotation UI + training trigger
├── train.py                 # YOLOv8 custom model training pipeline
├── rules.py                 # Stateless alert handler (photo save + log write)
├── record_training.py       # Utility: record labelled training videos from live cameras
├── preview_cameras.py       # Utility: show all cameras in a grid
├── run.sh                   # One-command launcher (kills old, starts both services)
├── requirements.txt         # Python pip dependencies
├── .env                     # Active config (credentials, thresholds) — NOT committed
├── .env.example             # Template
├── cameras.json             # Multi-camera config (4 active: D2, D6, D9, D19)
├── cameras.json.example     # Example with 4 different camera brands
├── .gitignore               # Excludes .env, *.pt, __pycache__, .claude/
├── project.md               # This file — full technical specification
│
├── yolov8s.pt               # COCO YOLOv8 Small (22.5 MB) — primary detection
├── yolov8n-pose.pt          # YOLOv8 Nano Pose (6.8 MB) — keypoint detection
├── yolov8n.pt               # YOLOv8 Nano (6.5 MB) — available, not used
├── yolo26n.pt               # Nano variant (5.5 MB) — available, not used
│
├── custom_model/
│   └── weights/
│       └── best.pt          # Custom-trained model (22.5 MB) — produced by train.py
│
├── training_data/
│   ├── dataset.yaml         # YOLOv8 dataset config (5 classes, train/val paths)
│   ├── split.json           # Fixed train/val split (preserved across retrains)
│   ├── images/              # Raw annotated images (85 total, various cameras)
│   │   ├── train/           # Symlinked/copied by train.py prepare_dataset()
│   │   └── val/             # Truly unseen — never augmented
│   ├── labels/              # YOLO format labels (59 label files)
│   │   ├── train/
│   │   └── val/
│   ├── augmented/           # Auto-generated offline augmented images
│   │   ├── images/
│   │   └── labels/
│   └── verified/            # OllamaVerifier-confirmed true positive alert photos
│
├── logs/
│   ├── logs.txt             # Alert log: [ts] [cam] [event] duration — desc | photo: path
│   ├── monitor.log          # stdout from monitor.py (piped by run.sh)
│   ├── web_ui.log           # stdout from web_ui.py / gunicorn
│   ├── train.log            # stdout from train.py subprocess
│   └── photos/              # Annotated alert JPEGs (1280×720)
│       └── Cam{N}_{EVENT}_{YYYYMMDD_HHMMSS}.jpg
│
└── runs/
    └── detect/
        └── val/             # YOLOv8 validation curves + confusion matrix PNGs
```

---

## 3. Active Camera Configuration (`cameras.json`)

Four Dahua cameras on one NVR at `192.168.30.5`:

```json
[
  { "id": 2,  "name": "D2",  "ip": "192.168.30.5", "user": "admin",
    "pass": "Masters@6677", "port": 554, "channel": 2,  "type": "dahua", "subtype": 0, "rtsp_path": "" },
  { "id": 6,  "name": "D6",  "ip": "192.168.30.5", "user": "admin",
    "pass": "Masters@6677", "port": 554, "channel": 6,  "type": "dahua", "subtype": 0, "rtsp_path": "" },
  { "id": 9,  "name": "D9",  "ip": "192.168.30.5", "user": "admin",
    "pass": "Masters@6677", "port": 554, "channel": 9,  "type": "dahua", "subtype": 0, "rtsp_path": "" },
  { "id": 19, "name": "D19", "ip": "192.168.30.5", "user": "admin",
    "pass": "Masters@6677", "port": 554, "channel": 19, "type": "dahua", "subtype": 0, "rtsp_path": "" }
]
```

**RTSP URL construction:**
- Dahua: `rtsp://admin:{url_pass}@{ip}:{port}/cam/realmonitor?channel={ch}&subtype={subtype}`
- Hikvision: `rtsp://admin:{url_pass}@{ip}:{port}/Streaming/Channels/{ch}0{subtype+1}`
- Custom: set `rtsp_path` field → overrides auto-build

`subtype=0` = main stream (higher res, used here). `subtype=1` = sub-stream (allows more concurrent NVR connections).

**Fallback when no cameras.json:** reads single camera from `.env` keys: `CAMERA_IP`, `CAMERA_USER`, `CAMERA_PASS`, `CAMERA_PORT`, `CAMERA_CHANNEL`, `CAMERA_TYPE`, `CAMERA_RTSP_PATH`.

---

## 4. Environment Configuration (`.env`)

```env
CAMERA_IP=192.168.30.5
CAMERA_USER=admin
CAMERA_PASS=Masters@6677
CAMERA_PORT=554
CAMERA_CHANNEL=2
CAMERA_RTSP_PATH=
CAMERA_TYPE=dahua

CHECK_INTERVAL_SEC=5
DETECTION_CONF=0.40

SLEEP_THRESHOLD_SEC=120        # 2 min — head-down before SLEEPING alert
PHONE_SESSION_GRACE_SEC=6      # seconds without detection = phone session ended
PHONE_SESSION_MIN_SEC=20       # ignore phone sessions shorter than this
PHONE_SESSION_MAX_SEC=600      # periodic save every N sec during long sessions

CAMERA_ANGLE=overhead
EAR_CONF_VISIBLE=0.55
EAR_CONF_HIDDEN=0.20
STANDING_RATIO=1.3
LEANING_BACK_RATIO=2.0

USE_OLLAMA=true
OLLAMA_MODEL=llava:latest
HEADLESS=true
```

---

## 5. Requirements (`requirements.txt`)

```
ultralytics>=8.0.0
opencv-python>=4.8.0
python-dotenv>=1.0.0
numpy>=1.24.0
requests>=2.31.0
flask>=3.0.0
```

**Optional (install separately):**
- `insightface` + `onnxruntime-gpu` — FaceWorker face detection (currently CPU only)
- `gunicorn` — production WSGI server (run.sh uses it if available, else flask dev server)
- `pyyaml` — required by train.py for dataset.yaml writing
- `scipy` — required by KalmanTracker for Hungarian assignment (available via ultralytics)

---

## 6. `monitor.py` — Complete Technical Specification (v3)

### 6.1 Imports and Structured Logging

```python
import logging
import os, time, signal, threading, json, urllib.parse, math
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2, torch, numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-14s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
_log = logging.getLogger("monitor")
```

All key events (model load, camera connect, alert, error) go through `_log` instead of `print()`. This allows filtering by level (`logging.WARNING` silences info in production) and provides ISO timestamps on every line.

### 6.2 Module-Level Constants

```python
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
HALF        = DEVICE == "cuda"          # fp16 on GPU

CONF_PERSON = 0.50    # YOLO person detection threshold
CONF_PHONE  = 0.35    # phone crop detection threshold
CONF_CHAIR  = 0.40    # chair (COCO 56) — used in future, not current alert
CONF_KP     = 0.50    # keypoint confidence minimum

DISPLAY_W   = 1280
DISPLAY_H   = 720

PERSON, PHONE, CHAIR = 0, 67, 56    # COCO class IDs

CUSTOM_PHONE_HAND = 0   # custom model class IDs
CUSTOM_PHONE_EAR  = 1
CUSTOM_PHONE_DESK = 2
CUSTOM_SLEEPING   = 3

SLEEP_THRESHOLD      = int(os.getenv("SLEEP_THRESHOLD_SEC",      "120"))
PHONE_SESSION_GRACE  = float(os.getenv("PHONE_SESSION_GRACE_SEC", "6"))
PHONE_SESSION_MIN    = float(os.getenv("PHONE_SESSION_MIN_SEC",   "20"))
PHONE_SESSION_MAX    = int(os.getenv("PHONE_SESSION_MAX_SEC",     "600"))

MOTION_STILL_SECS = 60
MOTION_THRESH     = 0.012    # EMA pixel-diff score < this → person is still
MIN_TRACK_AGE     = 30       # not currently used in alerts

CAMERA_ANGLE       = os.getenv("CAMERA_ANGLE",       "side").lower()
EAR_CONF_VISIBLE   = float(os.getenv("EAR_CONF_VISIBLE",   "0.50"))
EAR_CONF_HIDDEN    = float(os.getenv("EAR_CONF_HIDDEN",    "0.25"))
STANDING_RATIO     = float(os.getenv("STANDING_RATIO",     "1.6"))
LEANING_BACK_RATIO = float(os.getenv("LEANING_BACK_RATIO", "1.5"))
```

### 6.3 Shared GPU Models (`_load_models`)

Three models loaded ONCE at startup, shared across all cameras via `_infer_lock`:

```python
_detect_model  = None   # yolov8s.pt  — COCO person(0) + phone(67)
_custom_model  = None   # custom_model/weights/best.pt — phone_hand/ear/desk/sleeping
_pose_model    = None   # yolov8n-pose.pt
_infer_lock    = threading.Lock()    # serializes ALL GPU inference
```

Load sequence:
1. `yolov8s.pt` → warm-up dummy 416×416 frame on GPU
2. `custom_model/weights/best.pt` if exists → warm-up 320×320
3. `yolov8n-pose.pt` → warm-up 416×416

### 6.4 `EvidenceAcc` — Temporal Voting Accumulator

```python
class EvidenceAcc:
    def __init__(self, window=8, min_ratio=0.70, min_frames=None):
        self._h = deque(maxlen=window)
        self._r = min_ratio
        self._min = min_frames if min_frames is not None else max(3, window // 2)

    def add(self, v): self._h.append(1 if v else 0)
    def reset(self): self._h.clear()

    @property
    def confirmed(self):
        if len(self._h) < self._min: return False
        return (sum(self._h) / len(self._h)) >= self._r

    @property
    def ratio(self): return sum(self._h)/len(self._h) if self._h else 0.0
```

| Accumulator | window | min_ratio | min_frames | Notes |
|-------------|--------|-----------|------------|-------|
| `ev_phone` | 10 | 0.30 | 3 | 3/10 frames needed; custom model gives 20-38% conf overhead |
| `ev_sleep` | 12 | 0.75 | 6 (default) | 9/12 frames needed for sleep confirmation |

### 6.5 `TrackState` — Per-Person State

```python
class TrackState:
    def __init__(self, tid):
        self.tid = tid
        self.bbox = None                  # current (x1,y1,x2,y2)
        self.centroid = (0, 0)
        self.last_seen = time.time()
        self.first_seen = time.time()

        # Phone detection state
        self.has_phone = False            # YOLO detected phone in hand/ear
        self.phone_on_ear = False         # PoseWorker: wrist near ear keypoint
        self.phone_type = None            # 'hand' | 'ear' | 'desk' | None
        self.phone_bbox = None            # full-frame bbox of detected phone

        # Pose/motion/face state
        self.pose_sleeping = False
        self.face_visible = True
        self.motion_score = 1.0
        self.is_still = False
        self.still_since = None
        self.head_yaw = 0.0

        # Wrist keypoints (from PoseWorker)
        self.left_wrist  = None           # (x, y, conf) or None
        self.right_wrist = None

        # Evidence accumulators
        self.ev_phone = EvidenceAcc(window=10, min_ratio=0.30, min_frames=3)
        self.ev_sleep = EvidenceAcc(window=12, min_ratio=0.75)

        # Sleep session
        self.sleep_start       = None
        self.sleep_last_active = None
        self.sleep_session_ann = None

        # Phone session (photo saved when session ENDS)
        self.phone_session_start  = None
        self.phone_session_ptype  = None  # 'hand' or 'ear'
        self.phone_last_active    = None
        self.phone_session_ann    = None
        self.phone_session_saved  = 0.0
        self.phone_total_sec      = 0.0   # cumulative seconds across all sessions
```

**`sleep_raw` property:**
```python
@property
def sleep_raw(self):
    if not self.pose_sleeping: return False
    supporting = [
        not self.face_visible,
        self.is_still and self.still_since is not None
            and (time.time() - self.still_since) > MOTION_STILL_SECS,
    ]
    return sum(supporting) >= 1
```
`pose_sleeping` is REQUIRED. Then at least 1 supporting signal: face hidden OR body still for 60s.

**`phone_raw` property — v3 FIX:**
```python
@property
def phone_raw(self):
    yolo_detected = self.phone_type in ('hand', 'ear')
    # v3 FIX: phone_on_ear is now a STANDALONE signal.
    # Previously: pose_confirmed = self.phone_on_ear AND self.has_phone
    # Bug: from overhead, phone is hidden between hand+head → YOLO sees 0 phones
    # → has_phone was always False during a real call → every phone-call alert suppressed.
    # Fix: pose wrist-near-ear alone is sufficient; EvidenceAcc (min_frames=3)
    # and PHONE_SESSION_MIN=20s prevent brief head-touch false positives.
    pose_ear_only = self.phone_on_ear
    return yolo_detected or pose_ear_only
```

### 6.6 `CameraState` — Thread-Safe Per-Camera State

```python
class CameraState:
    def __init__(self):
        self._l = threading.Lock()
        self.frame   = None    # latest raw BGR numpy frame
        self.persons = []      # [{track_id, bbox, conf, centroid}, ...]
        self.phones  = []      # [{bbox, conf, track_id, type}, ...]
        self.alerts  = []      # overlay text list from BehaviorEngine

    def set_frame(self, f)      # thread-safe write
    def get_frame(self)         # thread-safe read
    def update(self, **kw)      # thread-safe attribute set
    def snapshot(self)          # returns dict copy (safe to iterate outside lock)
```

### 6.7 Tracker Utility — `_iou_matrix`

```python
def _iou_matrix(boxes_a: List[Tuple], boxes_b: List[Tuple]) -> np.ndarray:
    """Compute N×M IoU matrix between two lists of (x1,y1,x2,y2) boxes."""
```

Used by `KalmanTracker._associate()`. Returns float32 numpy array.

### 6.8 `KalmanBox` — Single-Object Kalman Filter (NEW in v3)

**State vector:** `X = [cx, cy, w, h, vx, vy]ᵀ` (6D, constant-velocity model)  
**Measurement:** `Z = [cx, cy, w, h]ᵀ` (4D — velocity is latent)

**Noise parameters (tuned for overhead office cameras at ~25fps):**
```python
_NOISE_POS  = 3.0    # position process noise  (px/frame)
_NOISE_VEL  = 8.0    # velocity process noise  (px/frame²)
_NOISE_MEAS = 6.0    # YOLO measurement noise  (px)
```

**Matrices:**
```
F (state transition):
  [1 0 0 0 1 0]   cx += vx
  [0 1 0 0 0 1]   cy += vy
  [0 0 1 0 0 0]   w unchanged
  [0 0 0 1 0 0]   h unchanged
  [0 0 0 0 1 0]   vx constant
  [0 0 0 0 0 1]   vy constant

H (measurement):  identity 4×6 (first 4 state dims)

Q (process noise):
  diag([p², p², p²×0.5, p²×0.5, v², v²])  where p²=9, v²=64

R (measurement noise):
  diag([r², r², r²×3, r²×3])  where r²=36
  (w/h jitter is larger than cx/cy jitter)

P (initial covariance):
  diag([4p², 4p², 8p², 8p², 50v², 50v²])
  high uncertainty on velocity, moderate on position
```

**`hit_streak` starts at 1** (track born from real detection = counts as first hit).

**Methods:** `predict()` → advances state, returns `(x1,y1,x2,y2)`. `update(bbox)` → Kalman correction step. `get_state()` → returns smoothed `(x1,y1,x2,y2)`.

### 6.9 `KalmanTracker` — Production Multi-Object Tracker (NEW in v3, replaces SimpleTracker)

**Drop-in replacement interface:** `tracker.update([(bbox, conf), ...]) → [(bbox, conf, track_id), ...]`

```python
class KalmanTracker:
    def __init__(self, max_age=45, min_hits=1, iou_threshold=0.20):
        self._next_id : int                   = 1
        self._boxes   : Dict[int, KalmanBox] = {}
        self._confs   : Dict[int, float]     = {}
```

**`update()` — 6-step algorithm:**
1. Predict all active tracks (advances each KalmanBox one frame)
2. Associate predictions ↔ detections via Hungarian algorithm + IoU gate
3. Update matched tracks (Kalman correction step)
4. Spawn new KalmanBox for every unmatched detection
5. Remove tracks where `time_since_update > max_age` (= 45 frames ≈ 22s)
6. Return only tracks that were matched this frame AND `hit_streak >= min_hits`

**Assignment method:**
- Primary: `scipy.optimize.linear_sum_assignment` (Hungarian algorithm — globally optimal)
- Fallback if scipy unavailable: `_greedy_associate` (same logic as old SimpleTracker)
- IoU gate: assignments with IoU < `iou_threshold=0.20` are rejected

**Parameters in `CameraSession`:**
```python
self.tracker = KalmanTracker(max_age=45, min_hits=1, iou_threshold=0.20)
```

**Why KalmanTracker beats SimpleTracker:**
- Kalman prediction keeps predicted bbox close to true position during movement → IoU stays high → no ID switch
- Hungarian gives globally optimal assignment (greedy can get stuck in local minima with 3+ persons)
- Smoothed bboxes reduce jitter in overlays and red-box photo placement
- `max_age=45` is identical to old SimpleTracker's `age < 45` removal policy → backward compatible

**SimpleTracker is kept in the file but NOT wired into CameraSession.** It exists for reference and fallback only.

### 6.10 `CameraSession`

```python
class CameraSession:
    def __init__(self, cfg):
        self.cam_id  = cfg['id']
        self.name    = cfg.get('name', f"Cam-{cfg['id']}")
        self.cfg     = cfg
        self.state   = CameraState()
        self.tracks  : Dict[int, TrackState] = {}
        self.tracks_lock = threading.Lock()
        self.tracker = KalmanTracker(max_age=45, min_hits=1, iou_threshold=0.20)

    def rtsp_url(self):
        # Dahua: /cam/realmonitor?channel={ch}&subtype={subtype}
        # Hikvision: /Streaming/Channels/{ch}0{subtype+1}
        # Password is URL-encoded with urllib.parse.quote(pass, safe='')
```

### 6.11 `BatchDetectWorker` — Two-Pass GPU Detection

**Thread:** `"BatchDetect"` | **Interval:** 0.5s target | **Shared across all cameras**

**Pass 1 — person detection on all camera frames (batched):**
```python
r1 = _detect_model(frames, verbose=False, conf=CONF_PERSON,
                   imgsz=416, device=DEVICE, half=HALF, classes=[PERSON])
```
- Input: list of raw BGR frames (one per live camera)
- Per person: runs through KalmanTracker, creates/updates TrackState
- Collects person crops with 25px padding for Pass 2

**Pass 2a — custom model on person crops (if custom model exists):**
```python
r2_custom = _custom_model(crops, verbose=False, conf=CONF_PHONE,
                          imgsz=320, device=DEVICE, half=HALF,
                          classes=[CUSTOM_PHONE_HAND, CUSTOM_PHONE_EAR, CUSTOM_PHONE_DESK])
```
- `skip_valid=True` — custom bboxes are person-sized (annotations were person-sized)
- Maps `cls_id → 'hand'|'ear'|'desk'`
- `hand` or `ear` NEVER downgraded to `desk`

**Pass 2b — COCO base model on same crops (always runs):**
```python
r2_coco = _detect_model(crops, verbose=False, conf=0.25,
                        imgsz=320, device=DEVICE, half=HALF, classes=[PHONE])
```
- Fills gaps missed by custom model
- Runs `_classify_phone()` to determine type: `'ear'|'hand'|'desk'`
- `_valid_phone()` filter applied (aspect ratio, area, position checks)

**Phone merging:** Custom model results added first; COCO results skip any track already having `hand`/`ear` classification.

**After each batch:** Tracks NOT seen in phones list this frame → `has_phone=False`, `phone_type=None`, `phone_bbox=None`.

**Track purge:** TrackState entries with `last_seen > 900s` removed each batch.

### 6.12 `_classify_phone()` — Phone Position Classifier

```python
def _classify_phone(phone_bbox, person_bbox, phone_on_ear_signal=False,
                    left_wrist=None, right_wrist=None) → 'ear'|'hand'|'desk':
```

Priority order:
1. `phone_on_ear_signal=True` → `'ear'`
2. Phone center in top 28% of person bbox → `'ear'`
3. Wrist keypoint within `MARGIN=30px` of phone bbox (conf ≥ 0.30) → `'hand'`
4. Default → `'desk'` (phone not being held)

### 6.13 `_valid_phone()` — Phone Geometry Filter

```python
def _valid_phone(phone_bbox, person_bbox, frame_h, frame_w) → bool:
```

| Check | Value | Rejects |
|-------|-------|---------|
| Portrait aspect | 1.3 ≤ h/w ≤ 4.5 | square objects |
| Landscape aspect | 0.22 ≤ h/w ≤ 0.77 | square objects |
| Min area | > 0.03% of frame | mice, pens |
| Max area | < 4% of frame | monitors, whiteboards |
| Position in person | 5%–95% of bbox height | phones above head or below feet |

### 6.14 `BatchPoseWorker` — Keypoint Estimation

**Thread:** `"BatchPose"` | **Interval:** 1.5s target | **Shared across all cameras**

```python
results = _pose_model(list(frames), verbose=False, conf=CONF_PERSON,
                      imgsz=416, device=DEVICE, half=HALF)
```

**COCO 17-keypoint indices used:**
- 0: nose, 3: left ear, 4: right ear
- 5: left shoulder, 6: right shoulder
- 9: left wrist, 10: right wrist

**Person matching:** nose keypoint must fall inside a person bbox from `cam.state.snapshot()['persons']`.

**Sleeping detection:**
```python
avg_sh = mean(shoulder y-coords where conf >= CONF_KP)
sh_span = max(|rs_x - ls_x|, 40)  # min clamp 40px
sleeping = nc >= CONF_KP and avg_sh is not None
           and nose_y > avg_sh + sh_span * 0.4
```
Nose below shoulders by 40% of shoulder span = head down = sleeping.

**Phone-on-ear detection — v3 FIX:**

**Old code (broken):**
```python
thr_y = sh_span * 0.55   # = 22px when sh_span=40 (clamped minimum)
thr_x = sh_span * 0.65   # = 26px when sh_span=40
```
**Root cause of bug:** From overhead cameras, shoulders are invisible (seen from above, shoulder width in 2D → near zero). `sh_span` always hits the 40px minimum. This gave `thr_y=22px, thr_x=26px` — so tight that a wrist 27px from ear would fail.

**New code (v3):**
```python
# Frame-relative minimums scale correctly at any camera resolution
thr_y = max(sh_span * 0.55, h * 0.06)   # 22px@360, 43px@720, 65px@1080
thr_x = max(sh_span * 0.65, w * 0.08)   # 51px@640, 102px@1280, 154px@1920
```
Validated at all resolutions: phone caller (wrist 57px from ear at 1080p) correctly triggers; non-callers (wrist 86px+ from ear) correctly rejected.

```python
if lwc >= C and lec >= C:
    if abs(lw_y - le_y) < thr_y and abs(lw_x - le_x) < thr_x:
        ear_phone = True
if rwc >= C and rec >= C:
    if abs(rw_y - re_y) < thr_y and abs(rw_x - re_x) < thr_x:
        ear_phone = True
```

**Head yaw:**
```python
yaw_deg = ((nose_x - ear_mid) / ear_span) * 90   # both ears visible
# Only left ear: yaw = +55.0    Only right ear: yaw = -55.0    None: yaw = 0.0
```

**Wrist output (for phone classifier):** returned if `wrist_conf >= 0.20`.

### 6.15 `MotionWorker` — CLAHE+EMA Motion Detection (IMPROVED in v3)

**Thread:** `"Motion-{cam_id}"` | **Interval:** 0.5s | **Per camera**

**v2 bug:** Raw `cv2.absdiff` on BGR grayscale — sensitive to monitor flicker, JPEG compression artifacts, auto-exposure shifts → false `is_still=False` signals → false sleep detection misses.

**v3 improvements:**
1. **CLAHE normalization** on LAB L-channel: removes uniform lighting changes (monitor flicker, auto-exposure)
2. **Gaussian blur** (3×3) on diff: removes JPEG block noise and minor camera tremor
3. **EMA smoothing** (α=0.35): dampens single-frame spikes from compression artifacts
4. **Stale-ID cleanup**: removes entries from `_prev` and `_scores` dicts when track disappears (prevents memory leak)

```python
class MotionWorker(threading.Thread):
    _EMA_ALPHA = 0.35

    def __init__(self, cam):
        self._prev  : Dict[int, np.ndarray] = {}   # tid → 64×64 L-channel
        self._scores: Dict[int, float]      = {}   # tid → EMA score
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    def _process(self, frame):
        for pe in persons:
            crop = frame[y1:y2, x1:x2]
            small = cv2.resize(crop, (64, 64))
            lab   = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
            norm  = self._clahe.apply(cv2.split(lab)[0])   # normalized luminance

            raw_score = 1.0
            if tid in self._prev:
                diff = cv2.absdiff(norm, self._prev[tid])
                diff = cv2.GaussianBlur(diff, (3, 3), 0)
                raw_score = float(diff.mean()) / 255.0
            self._prev[tid] = norm

            prev_ema = self._scores.get(tid, raw_score)
            ema_score = 0.35 * raw_score + 0.65 * prev_ema
            self._scores[tid] = ema_score

            ts.motion_score = ema_score
            ts.is_still     = (ema_score < MOTION_THRESH)   # MOTION_THRESH = 0.012

        # Cleanup stale TIDs — prevents unbounded growth
        stale = [k for k in self._prev if k not in active_tids]
        for k in stale:
            self._prev.pop(k, None)
            self._scores.pop(k, None)
```

**CLAHE performance:** Reduces false motion from a uniform +60 brightness change by ~54% on real textured footage.

### 6.16 `FaceWorker` — InsightFace Face Visibility

**Thread:** `"Face-{cam_id}"` | **Interval:** 2.0s | **Per camera**

```python
self._fa = insightface.app.FaceAnalysis(
    name="buffalo_sc",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    allowed_modules=["detection"])   # only detection, NOT recognition
self._fa.prepare(ctx_id=0, det_size=(160, 160))
```

Runs face detection on each person crop (160×160). Sets `ts.face_visible = len(faces) > 0`. If InsightFace unavailable: silently disables, `face_visible` stays `True` (conservative — doesn't generate false sleep FPs).

### 6.17 `OllamaVerifier` — False Positive Removal (MEMORY LEAK FIXED in v3)

**Thread:** `"OllamaVerifier"` | **Interval:** 8s | **Global single thread**

**v2 bug:** `self._seen` set grew unboundedly — filenames of deleted photos were never removed, so any future photo with the same timestamp string would be silently skipped.

**v3 fix:**
```python
# Prune _seen before each scan
existing_names = {p.name for p in self._photos.glob("Cam*.jpg")}
self._seen &= existing_names   # intersection update — removes deleted filenames
```

**Scan logic:** Takes 15 most recent `Cam*.jpg` by mtime. For each unseen photo, determines event from filename, sends to Ollama llava with event-specific question. Answer:
- `"NO"` → delete photo, remove from `logs/logs.txt`
- `"YES"` → copy to `training_data/verified/` for future training

**Ollama API call:**
```python
POST http://localhost:11434/api/generate
{"model": OLLAMA_MODEL, "prompt": question, "images": [base64], "stream": False,
 "options": {"temperature": 0.05, "num_predict": 6}}
```
Temperature=0.05 (near-deterministic). First word of response parsed: `YES` or `NO`. Uncertain/error → keep photo (default YES).

**Three questions (one per event type):**
- `PHONE_HAND`: Asks if person in red/orange box is CLEARLY holding a phone (not mouse, pen, keyboard)
- `PHONE_EAR`: Asks if person is CLEARLY holding phone to ear while talking (not touching face/head)
- `SLEEPING`: Asks if person's head is resting DOWN on desk appearing asleep

### 6.18 `BehaviorEngine` — Alert State Machine

**Thread:** `"Behavior-{cam_id}"` | **Interval:** 0.5s | **Per camera**

**Critical guard:** Only fires alerts for persons currently visible (`live_bbox_by_tid.get(tid)` check). Stale tracks (person left) never produce alerts.

**Phone session state machine:**
```
phone_ok=True (ev_phone.confirmed):
  if phone_session_start is None:
    → NEW SESSION: save IMMEDIATE photo at t=0s
    → fire alert immediately
  else:
    → ONGOING: update best frame every 8s
  
  Periodic save: if session_dur ≥ PHONE_SESSION_MAX and
                    now - session_saved ≥ PHONE_SESSION_MAX → periodic alert

phone_ok=False:
  if phone_session_start is not None and
     now - phone_last_active ≥ PHONE_SESSION_GRACE (6s):
    → SESSION ENDED: fire end-of-session alert with full duration
    → only if duration ≥ PHONE_SESSION_MIN (20s)
    → reset all session state
    → ts.phone_total_sec += duration
```

**Sleep session state machine:**
```
sleep_ok=True:
  if sleep_start is None: sleep_start = now
  elapsed = now - sleep_start
  if elapsed ≥ SLEEP_THRESHOLD (120s) and sleep_session_ann is None:
    → PHOTO 1: immediate alert with elapsed duration

sleep_ok=False:
  if sleep_start is not None and sleep_session_ann is not None:
    duration = sleep_last_active - sleep_start
    if duration ≥ SLEEP_THRESHOLD:
      → PHOTO 2: wake-up alert with exact total duration
  → reset sleep state
```

**`_force_red_box(ann_frame, ts, ptype, orig_frame)`:**
Forces a red rectangle + label on the violator in saved alert photos, using `ts._live_bbox` (current detection position, not stale tracker bbox).

### 6.19 `camera_reader` — RTSP Reader (IMPROVED in v3)

**Startup delay:** cameras staggered 1.5s each (0s, 1.5s, 3.0s, 4.5s) to avoid simultaneous NVR connection burst.

**v3 improvements:**
1. **Exponential backoff** on connect failure: `retry_delay = min(delay × 1.5, 30.0)` — was linear +2s/attempt, max 15s
2. **Consecutive-fail guard:** requires 5 consecutive bad reads before reconnecting — absorbs transient RTSP packet drops without unnecessary reconnects

```python
_RETRY_INIT     = 2.0   # initial retry delay seconds
_RETRY_MULT     = 1.5   # backoff multiplier
_RETRY_MAX      = 30.0  # maximum retry delay
_FAIL_THRESHOLD = 5     # consecutive bad reads before reconnect

# On connect success: retry_delay resets to _RETRY_INIT
# On read failure: consecutive_fails++; if >= _FAIL_THRESHOLD → reconnect
```

**Frame sharing with web_ui:**
```python
# Every 3rd frame (≈3fps write rate)
shared = cv2.resize(frame, (640, 360))
tmp = SHARED_FRAMES_DIR / f"{cam.name}.tmp.jpg"
cv2.imwrite(str(tmp), shared, [cv2.IMWRITE_JPEG_QUALITY, 72])
tmp.rename(SHARED_FRAMES_DIR / f"{cam.name}.jpg")   # atomic rename
```

`SHARED_FRAMES_DIR = Path("/tmp/monitor_frames")` — zero additional RTSP connections for web_ui.

### 6.20 Annotation — `annotate_cam(cam, target_w=None, target_h=None)`

**Person box colors:**
- Green `(0,200,0)` = OK / working normally
- Red `(0,0,220)` = PHONE IN HAND or SLEEPING
- Deep orange-red `(0,60,255)` = PHONE ON EAR

**Phone object box colors:**
- `(0,60,255)` = phone on ear
- `(100,100,100)` = phone on desk (grey, not an alert)
- `(0,140,255)` = phone in hand (orange)

**Status bar:** Bottom 26px dark bar: `"{cam.name} | HH:MM:SS | Persons:{n} Phone:{n} Sleeping:{n}"`

### 6.21 `main()` Startup Sequence

1. Print GPU info (device, VRAM)
2. `_load_models()` — all three models
3. `_load_camera_configs()` — from cameras.json or .env
4. Start RTSP reader threads (staggered 1.5s each)
5. Wait up to 15s for first live frame
6. Start `BatchDetectWorker` + `BatchPoseWorker` (shared)
7. Per camera: start `MotionWorker`, `FaceWorker`, `BehaviorEngine`
8. Start `OllamaVerifier`
9. `HEADLESS=true`: `while _running: time.sleep(1)`

---

## 7. `rules.py` — Alert Handler (PATHS FIXED in v3)

### 7.1 Path Fix

```python
# v3: absolute paths — never breaks when CWD differs from project root
_BASE      = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR   = os.path.join(_BASE, "logs")
PHOTOS_DIR = os.path.join(LOGS_DIR, "photos")
LOGS_FILE  = os.path.join(LOGS_DIR, "logs.txt")

os.makedirs(PHOTOS_DIR, exist_ok=True)   # created at import time
os.makedirs(LOGS_DIR,   exist_ok=True)
```

### 7.2 `fire_alert(event, frame, det, elapsed_sec, cam_id, cam_name)`

1. Generates filename: `Cam{cam_id}_{EVENT}_{YYYYMMDD_HHMMSS}.jpg`
2. Calls `_draw()` — adds banner to already-annotated frame
3. `cv2.imwrite(photo_path, annotated_frame)` — saves 1280×720 JPEG
4. Appends to `logs/logs.txt`: `[ts] [Cam{N}:{name}] [{EVENT}] {m}m {s:02d}s — {msg} | photo: {path}`
5. Prints red ANSI `[ALERT]` line to stdout
6. If `USE_OLLAMA=true`: spawns background thread → `_ollama_update()`

### 7.3 `_draw()` — Photo Banner

Adds event-specific colored banner (full width, 80px height) at top of frame:
- `PHONE_HAND`: dark blue `(20, 90, 200)`
- `PHONE_EAR`: dark blue-red `(0, 60, 220)`
- `SLEEPING`: dark purple `(40, 40, 160)`

Banner text: `"  [{cam_name}]  {LABEL}   |   Total: {m} min {s:02d} sec"`

Timestamp bottom-right: `"YYYY-MM-DD  HH:MM:SS"`

### 7.4 Event Labels

| Event | Label shown | Default log message |
|-------|-------------|---------------------|
| PHONE_HAND | `PHONE IN HAND` | Employee detected holding phone in hand. |
| PHONE_EAR | `PHONE ON EAR / CALLING` | Employee detected talking on phone / phone held to ear. |
| SLEEPING | `SLEEPING / HEAD ON DESK` | Employee appears sleeping — head down, no face visible, or motionless. |

---

## 8. `web_ui.py` — Flask Web Dashboard (IMPROVED in v3)

### 8.1 Path Fix (v3)

```python
_BASE      = Path(__file__).parent.resolve()     # always project root
PHOTOS_DIR = _BASE / "logs" / "photos"
LOGS_FILE  = _BASE / "logs" / "logs.txt"
PHOTOS_DIR.mkdir(parents=True, exist_ok=True)    # created at startup
CAMS = json.load(open(_BASE / "cameras.json")) if (_BASE / "cameras.json").exists() else []
```

### 8.2 Stale Frame Detection (v3 FIX)

**v2 bug:** `CameraStream._poll` only set `connected=False` when the shared frame file was deleted. Stale files from a stopped monitor showed all cameras as "Live" (green LED) indefinitely.

**v3 fix:**
```python
STALE_SEC = 5.0

def _poll(self):
    last_mtime = 0
    while True:
        if self._file.exists():
            mt  = self._file.stat().st_mtime
            now = time.time()
            if mt != last_mtime:
                # Load new frame
                self.connected = True
                last_mtime = mt
            else:
                # File not updated — mark disconnected if stale
                self.connected = (now - mt) < STALE_SEC
        else:
            self.connected = False
        time.sleep(0.15)
```

Cameras go to red LED within 5s of monitor.py stopping.

### 8.3 Architecture

- **No RTSP connections in web_ui.** Only reads `/tmp/monitor_frames/{name}.jpg` files written by monitor.py.
- `CameraStream._poll`: checks file mtime every 150ms
- **Snapshot polling** (not MJPEG): browser polls `/snapshot/{name}` every 600ms, staggered 85ms between cameras

### 8.4 Routes

| Route | Description |
|-------|-------------|
| `GET /` | Dashboard HTML |
| `GET /annotate` | Training annotation page |
| `GET /snapshot/<name>` | 640×360 JPEG from shared frame |
| `GET /snapshot/<name>/hd` | 1280×720 JPEG (for modal + annotation) |
| `GET /photos/<filename>` | Serve alert photo from `logs/photos/` |
| `GET /api/status` | Camera connected dict |
| `GET /api/alerts` | Recent alert sessions (cached 4s) |
| `GET /api/camera/<name>/alerts` | Per-camera alerts (cached 4s) |
| `GET /api/logs` | Last 60 log entries (cached 5s) |
| `POST /api/annotate` | Save annotation (cam, label, box) |
| `GET /api/annotation_counts` | Per-class annotation counts |
| `POST /api/train` | Start train.py subprocess |
| `GET /api/train_status` | Training status: idle/running/done/error |

### 8.5 Alert Session Grouping

`_parse_photo(path)` — parses filename: `Cam{N}_{EVENT}_{YYYYMMDD}_{HHMMSS}.jpg`

`_sessions(photos, gap_min=4)` — groups consecutive same-cam+event photos within 4 minutes into sessions:
- `duration_str`: "X min YY sec" or "Y sec"
- `duration_label`: "{X}m {YY}s"

### 8.6 Training Classes

```python
CLASSES  = ["phone_hand", "phone_ear", "phone_desk", "sleeping", "working"]
CLASS_IDS = {c: i for i, c in enumerate(CLASSES)}
# 0: phone_hand, 1: phone_ear, 2: phone_desk, 3: sleeping, 4: working
```

"working" = negative sample → empty label file (no objects).

### 8.7 Training Trigger

- Minimum 15 images required
- `subprocess.Popen(["python3", "train.py"])` → stdout to `logs/train.log`
- Status polled at `/api/train_status` every 5s

---

## 9. `train.py` — Custom Model Training Pipeline (IMPROVED in v3)

### 9.1 Constants

```python
TRAIN_DIR        = Path("training_data")
AUG_DIR          = TRAIN_DIR / "augmented"
OUT_DIR          = Path("custom_model")
SPLIT_FILE       = TRAIN_DIR / "split.json"
CLASSES          = ["phone_hand", "phone_ear", "phone_desk", "sleeping", "working"]
MIN_IMAGES       = 15
TARGET_PER_CLASS = 60
```

### 9.2 Image Quality Filter

```python
def is_good_quality(img_path, min_blur=50, min_bright=20, max_bright=235):
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = gray.mean()
    return blur_score >= 50 and 20 <= brightness <= 235
```

### 9.3 Preprocessing — `preprocess_image(img)`

CLAHE on L channel (LAB space):
```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
```
Applied to every original image before augmentation. Handles mixed office lighting.

### 9.4 Offline Augmentation — 10 Operations

```python
AUGMENT_OPS = [
    "hflip",         # cv2.flip → labels: cx = 1 - cx
    "bright_up",     # +45 pixel value
    "bright_down",   # -40 pixel value
    "contrast_up",   # alpha=1.35, beta=-20
    "contrast_down", # alpha=0.70, beta=30
    "noise",         # Gaussian noise std=12
    "blur_slight",   # horizontal motion blur kernel
    "rot90",         # cv2.ROTATE_90_CLOCKWISE → labels rotated
    "hflip_bright",  # flip + +30
    "clahe_boost",   # strong CLAHE clipLimit=4.0
]
```

Label transforms: horizontal flip (`cx = 1-cx`), rot90 (`(cx,cy,bw,bh) → (1-cy, cx, bh, bw)`).

Priority: images with rare classes get more augmentation. Target: 60 annotations per class.

### 9.5 Fixed Train/Val Split

`val_ratio=0.18`, minimum 2 val images. Split saved to `split.json` — never changes for existing images. New images always go to train. Augmented images ALWAYS train only (never val).

### 9.6 Auto Model Selection

```python
# existing best.pt → continue training from checkpoint
# n_train >= 200 → yolov8m.pt (medium)
# n_train < 200  → yolov8s.pt (small, frozen backbone)
```

### 9.7 Class Weight Computation (NEW in v3)

```python
def compute_class_weights(lbl_dir: Path) -> float:
    """
    Computes cls_pw scalar for YOLOv8 BCE classification loss.
    Formula: min(median_count / min_count, 5.0)
    Upweights rare classes to reduce false negatives on minority classes.
    """
    # Returns float capped at 5.0
    # Example: phone_hand=30, phone_ear=5, sleeping=10, working=30
    #   min=5, median=20 → cls_pw = 20/5 = 4.0
```

### 9.8 Training Hyperparameters

```python
model.train(
    data    = str(yaml_path),
    epochs  = 200 if n_train < 200 else 150,
    imgsz   = 640,
    batch   = 8 if n_train < 200 else 16,
    device  = "cuda" if torch.cuda.is_available() else "cpu",

    # Learning
    lr0     = 0.0003,    lrf     = 0.003,
    cos_lr  = True,      warmup_epochs = 5,
    warmup_momentum = 0.8,   momentum = 0.937,

    # Regularization
    weight_decay    = 0.001,
    label_smoothing = 0.1,
    dropout         = 0.1,

    # Class-weighted loss (NEW v3)
    cls_pw  = cls_pw,    # computed by compute_class_weights()

    # Frozen backbone
    freeze  = 15 if n_train < 150 else 10,

    # Online augmentation
    mosaic=1.0, copy_paste=0.2, mixup=0.08,
    degrees=12.0, translate=0.12, scale=0.55,
    shear=2.0, perspective=0.0003,
    fliplr=0.5, flipud=0.0,       # flipud=0: overhead camera has fixed orientation
    hsv_h=0.02, hsv_s=0.7, hsv_v=0.4,
    erasing=0.35, close_mosaic=30,

    # Control
    patience=40, project=str(OUT_DIR), name="weights",
    exist_ok=True, verbose=True, plots=True, save_period=10,
)
```

Output: `custom_model/weights/best.pt`

### 9.9 CLI

```bash
python3 train.py              # augment + train
python3 train.py --check      # dataset stats only
python3 train.py --augment    # augmentation only
python3 train.py --eval       # evaluate existing model
python3 train.py --no-augment # skip augmentation
```

---

## 10. `run.sh` — One-Command Launcher

1. `cd "$(dirname "$0")"` — change to project directory
2. Kill previous instances by PID + by path (project-specific only, does NOT kill other processes)
3. `pip install -q -r requirements.txt`
4. Set env: `QT_QPA_PLATFORM=xcb`, `HEADLESS=true`, clear log files
5. `python3 -u monitor.py >> logs/monitor.log 2>&1 &` → save PID to `/tmp/emp_monitor.pid`
6. Start gunicorn (if available) or flask dev server → save PID to `/tmp/emp_webui.pid`
7. Print dashboard URL with local IP
8. `tail -f logs/monitor.log` with noise filter (hides InsightFace/ONNX/HEVC warnings)
9. `trap cleanup SIGINT SIGTERM` — kills both on Ctrl+C

**Gunicorn config:** `--workers=1 --threads=50 --worker-class=gthread --bind=0.0.0.0:5000 --timeout=30 --keep-alive=2`

---

## 11. Thread Architecture

| Thread | Count | Interval | Purpose |
|--------|-------|----------|---------|
| `camera_reader` | 1 per camera | continuous | RTSP frame grab + shared file write |
| `BatchDetect` | 1 (shared) | 0.5s | Person + phone detection all cameras |
| `BatchPose` | 1 (shared) | 1.5s | Pose keypoints all cameras |
| `Motion-{id}` | 1 per camera | 0.5s | CLAHE+EMA pixel diff stillness |
| `Face-{id}` | 1 per camera | 2.0s | InsightFace face visibility |
| `Behavior-{id}` | 1 per camera | 0.5s | Alert state machine |
| `OllamaVerifier` | 1 (global) | 8s | LLaVA false-positive removal |
| `CameraStream._poll` | 1 per camera (web_ui) | 0.15s | Read shared frame files |

**For 4 cameras: 14 threads** (monitor.py) + 4 threads (web_ui.py CameraStream polls).  
**Shared state lock:** `_infer_lock` serializes all GPU calls.  
**Per-camera lock:** `cam.tracks_lock` protects TrackState dict.  
**Per-camera lock:** `cam.state._l` protects frame/persons/phones/alerts.

---

## 12. Model Details

### yolov8s.pt — Primary Detection
- Architecture: YOLOv8 Small (COCO pre-trained, 22.5MB)
- Used: person detection (class 0) Pass 1, phone detection (class 67) Pass 2
- Params: `imgsz=416, fp16, conf=0.50` (persons) / `conf=0.25` (phones on crops)
- Cannot be replaced by custom model — different class IDs

### yolov8n-pose.pt — Pose Estimation
- Architecture: YOLOv8 Nano Pose (COCO, 6.8MB)
- 17 keypoints; outputs in original frame coordinate space
- Params: `imgsz=416, fp16, conf=0.50`

### custom_model/weights/best.pt — Custom Trained
- Architecture: YOLOv8 Small/Medium, fine-tuned from COCO
- Classes: 0=phone_hand, 1=phone_ear, 2=phone_desk, 3=sleeping, 4=working
- Training data: 85 images, overhead angle, this specific office
- Inference: `imgsz=320, fp16, conf=0.35, skip_valid=True`
- Bboxes are person-sized (annotations drawn around person, not phone)
- Runs ALONGSIDE base model — both results merged

---

## 13. Detection Pipeline — End-to-End Flow

```
[Camera RTSP] → camera_reader thread
                 ↓ (every frame)
              cam.state.set_frame(frame)
                 ↓ (every 3rd frame, atomic)
              /tmp/monitor_frames/{name}.jpg → web_ui reads this

[BatchDetectWorker — every 0.5s]
  Pass 1: yolov8s(frames, imgsz=416, classes=[PERSON])
    → person bboxes → KalmanTracker (Hungarian + Kalman prediction)
    → creates/updates TrackState per person
    → collects person crops (25px padding)
  Pass 2a: custom_model(crops, imgsz=320) if exists
    → phone_hand/ear/desk → added first (skip_valid=True)
  Pass 2b: yolov8s(crops, imgsz=320, classes=[PHONE], conf=0.25)
    → _classify_phone() → _valid_phone() filter
    → merged; hand/ear never downgraded to desk
  → cam.state.update(persons=..., phones=...)

[BatchPoseWorker — every 1.5s]
  yolov8n-pose(frames, imgsz=416)
    → sleeping: nose_y > avg_shoulder_y + sh_span*0.4
    → phone_on_ear: wrist within thr_y=max(sh_span*0.55, h*0.06),
                               thr_x=max(sh_span*0.65, w*0.08) of ear
    → head_yaw, wrist positions
  → ts.pose_sleeping, ts.phone_on_ear, ts.head_yaw, ts.left/right_wrist

[MotionWorker — every 0.5s, per camera]
  CLAHE-normalized LAB L-channel crop (64×64)
  → GaussianBlur absdiff → EMA(α=0.35) smoothing
  → ts.motion_score, ts.is_still, ts.still_since

[FaceWorker — every 2.0s, per camera]
  InsightFace(160×160 person crop)
  → ts.face_visible

[BehaviorEngine — every 0.5s, per camera]
  ts.ev_phone.add(ts.phone_raw)   # phone_raw = YOLO OR pose_ear_only (v3 fix)
  ts.ev_sleep.add(ts.sleep_raw)   # sleep_raw = pose + (face_hidden OR still_60s)
  phone_ok = ev_phone.confirmed   # 3/10 frames = confirmed
  sleep_ok = ev_sleep.confirmed   # 9/12 frames = confirmed
  Only if person currently visible (live_bbox check):
    → Phone session state machine → rules.fire_alert()
    → Sleep session state machine → rules.fire_alert()

[OllamaVerifier — every 8s]
  Scans logs/photos/*.jpg
  → POST llava with YES/NO question
  → NO: delete photo + remove from logs.txt
  → YES: copy to training_data/verified/
```

---

## 14. Key Bug Fixes in v3

### Bug 1 — Phone-on-Ear Never Detected from Overhead Camera (CRITICAL)

**Problem:** `phone_raw` property required `has_phone=True` (YOLO must detect phone object) for the pose wrist-near-ear signal to count. From overhead, the phone is physically hidden between the hand and head during a call — YOLO detected zero phones at any confidence. Every overhead phone-call alert was suppressed.

**Fix:** `pose_ear_only = self.phone_on_ear` is now a standalone trigger. Brief head-touches don't false-positive because EvidenceAcc requires `min_frames=3` of sustained wrist-near-ear, and `PHONE_SESSION_MIN=20s` prevents saving photos from very short events.

### Bug 2 — Wrist-to-Ear Threshold Too Tight for Overhead Cameras

**Problem:** `thr_y = sh_span * 0.55` where `sh_span` always clamps to 40px from overhead (shoulders not visible from above in 2D) → `thr_y=22px, thr_x=26px`. A phone caller with wrist 27px from ear failed by 5px. Validated on live footage: caller had wrist-ear dy=57px at 1080p — would fail the old 22px threshold.

**Fix:** Frame-relative minimums: `thr_y = max(sh_span*0.55, h*0.06)`, `thr_x = max(sh_span*0.65, w*0.08)`. Scales correctly at 640×360 (22px/51px), 1280×720 (43px/102px), 1920×1080 (65px/154px). Validated: caller triggers at all resolutions; non-callers (wrist 86px+ from ear) correctly blocked by the y-threshold.

### Bug 3 — `logs/photos/` Relative Path Breaking Under Different CWDs

**Problem:** Both `rules.py` and `web_ui.py` used relative paths (`"logs/photos"`). If CWD at startup differed from project root, photos were saved/served from wrong locations. Photos directory didn't exist at startup → `api_alerts` returned `[]`.

**Fix:** Both files now use `__file__`-based absolute paths. `web_ui.py` creates the directory at import time.

### Bug 4 — Stale Camera Frames Showing "Live" After Monitor Stops

**Problem:** `CameraStream._poll` only set `connected=False` when the shared frame file was deleted. Old stale frames from a previous run kept all cameras showing as "Live" (green LED) permanently.

**Fix:** Added 5-second staleness check: `self.connected = (now - mt) < STALE_SEC`. Cameras go red within 5s of monitor.py stopping.

### Bug 5 — `OllamaVerifier._seen` Memory Leak

**Problem:** Filenames of deleted photos were never removed from `self._seen`. Over days, the set grew unboundedly. Also, if OllamaVerifier deleted a photo (answer=NO) and a later photo happened to get the same timestamp string, it would be silently skipped.

**Fix:** `self._seen &= {p.name for p in self._photos.glob("Cam*.jpg")}` — intersection update before each scan removes non-existent filenames.

### Bug 6 — `MotionWorker._prev` and `_scores` Memory Leak

**Problem:** When KalmanTracker assigned new track IDs (ID switch), old IDs accumulated in `_prev` and `_scores` dicts indefinitely.

**Fix:** After each `_process()` call, stale TIDs (not in current `active_tids`) are removed from both dicts.

### Bug 7 — RTSP Single Bad Read Triggered Immediate Reconnect

**Problem:** One failed `cap.read()` immediately broke the inner loop and started a reconnect. Transient RTSP packet drops caused unnecessary reconnects that spiked NVR load.

**Fix:** `_FAIL_THRESHOLD=5` consecutive bad reads required before reconnecting. Added exponential backoff (×1.5 per attempt, max 30s).

---

## 15. Training Dataset — Current State

| Item | Value |
|------|-------|
| Total images | 85 |
| Label files | 59 |
| Cameras represented | D1, D2, D6, D9, D19, D31, D44 |
| Camera angle | Overhead (ceiling-mounted) |
| Verified folder | `training_data/verified/` — OllamaVerifier-confirmed true positives |

**Classes:**
```yaml
names: [phone_hand, phone_ear, phone_desk, sleeping, working]
nc: 5
```

**Required for 90%+ real-world accuracy (per architecture audit):**
- phone_hand: 300+ instances (currently ~20)
- phone_ear: 150+ instances (currently ~10)
- phone_desk: 200+ hard negative instances (currently ~15)
- sleeping: 100+ instances (currently ~8)
- working: 400+ negative instances including hard negatives

---

## 16. Shared State — Thread Safety Reference

| Object | Lock | Writers | Readers |
|--------|------|---------|---------|
| `cam.state.frame` | `cam.state._l` | `camera_reader` | BatchDetect, BatchPose, Motion, Face, Behavior |
| `cam.state.persons/phones/alerts` | `cam.state._l` | BatchDetect, Behavior | BatchPose, Motion, Face, Behavior, annotate_cam |
| `cam.tracks` dict | `cam.tracks_lock` | BatchDetect, BatchPose, Motion, Face, Behavior | All of above |
| `_detect_model`, `_custom_model`, `_pose_model` | `_infer_lock` | loaded once | BatchDetect, BatchPose |
| `/tmp/monitor_frames/*.jpg` | atomic rename | camera_reader | web_ui CameraStream._poll |
| `logs/logs.txt` | Python file append | rules.fire_alert | web_ui /api/logs, OllamaVerifier |
| `logs/photos/*.jpg` | write-once | rules.fire_alert | web_ui /api/alerts, OllamaVerifier |

---

## 17. Pending Improvements (From Architecture Audit)

Ordered by priority (highest first):

| Priority | Item | Expected Gain |
|----------|------|---------------|
| 10 | Hard-negative mining automation | Phone precision +30–40% |
| 9 | Time-based confidence-weighted EvidenceAcc | F1 +15–20% |
| 9 | Event fusion engine (probabilistic multi-signal) | Phone precision +20–30% |
| 8 | Camera calibration profiles per-camera in cameras.json | Per-cam FP −20–40% |
| 8 | Desk zone employee re-ID (spatial centroid matching) | Session timing +25–35% |
| 7 | ByteTrack Lost-state extension (hibernate buffer) | ID stability +10–15% |
| 7 | Head pose pitch + roll (2D proxy from keypoints) | Sleep recall +10–12% |
| 6 | InsightFace recognition (face embeddings + gallery) | Session timing +20–30% |
| 5 | TensorRT export + motion-gated batching (20+ cameras) | Latency only |

---

## 18. How to Run

```bash
# Start everything
chmod +x run.sh
./run.sh

# Dashboard:   http://{local_ip}:5000
# Training UI: http://{local_ip}:5000/annotate

# Check logs
tail -f logs/monitor.log
tail -f logs/logs.txt

# View photos
ls logs/photos/

# Train custom model (after annotating 15+ images via /annotate)
python3 train.py

# Check dataset stats
python3 train.py --check
```

**To test phone detection:** Hold phone visibly to camera for 5-10 seconds. Alert fires within 3-5 seconds and photo appears in dashboard Recent Alerts panel within 4 seconds (API cache TTL).
