# Employee Monitor — Complete Project Blueprint

> This document is a full technical specification of the project. Every file, class, function, configuration value, hyperparameter, algorithm, and design decision is documented here. An LLM given only this file should be able to recreate the entire project exactly as it exists.

---

## 1. Project Overview

**Name:** Employee Monitor v2 — Multi-Camera, GPU-Efficient  
**Purpose:** Real-time AI-powered CCTV employee behaviour monitoring via RTSP streams. Detects phone usage, sleeping employees, and generates annotated photo evidence with duration tracking.  
**Language:** Python 3.8+  
**Primary entry point:** `monitor.py`  
**Web dashboard:** `web_ui.py` (Flask, port 5000)  
**Camera type tested:** Dahua NVR (multi-channel), Hikvision compatible  
**Hardware:** NVIDIA GPU strongly recommended; CPU fallback supported  
**Operating mode:** Always headless (no cv2 window); web browser is the UI

---

## 2. Directory Structure

```
employees/
├── monitor.py               # Main AI detection engine (multi-camera, GPU)
├── web_ui.py                # Flask web dashboard + annotation UI + training trigger
├── train.py                 # YOLOv8 custom model training pipeline
├── rules.py                 # Stateless alert handler (photo save + log write)
├── record_training.py       # Utility: record labelled training videos from live cameras
├── preview_cameras.py       # Utility: show all cameras in a grid to plan recording
├── run.sh                   # One-command launcher (kills old instances, starts both services)
├── requirements.txt         # Python pip dependencies
├── .env                     # Active environment config (credentials, thresholds) — NOT committed
├── .env.example             # Template for .env
├── cameras.json             # Multi-camera config (4 cameras active: D2, D6, D9, D19)
├── cameras.json.example     # Example cameras.json with 4 different camera brands
├── .gitignore               # Excludes .env, *.pt, __pycache__, .claude/
│
├── yolov8s.pt               # COCO YOLOv8 Small (22.5 MB) — primary detection model
├── yolov8n-pose.pt          # YOLOv8 Nano Pose (6.8 MB) — keypoint detection
├── yolov8n.pt               # YOLOv8 Nano (6.5 MB) — available, not used in v2
├── yolo26n.pt               # Another nano variant (5.5 MB) — available, not used
│
├── custom_model/
│   └── weights/
│       └── best.pt          # Trained custom model (22.5 MB) — saved by train.py
│
├── training_data/
│   ├── dataset.yaml         # YOLOv8 dataset config (5 classes, train/val paths)
│   ├── split.json           # Fixed train/val split (preserved across retrains)
│   ├── images/              # Raw annotated images (85 total, various cameras)
│   │   ├── train/           # Created by train.py prepare_dataset()
│   │   └── val/             # Created by train.py prepare_dataset()
│   ├── labels/              # YOLO format labels matching images/ (59 label files)
│   │   ├── train/
│   │   └── val/
│   ├── augmented/           # Auto-generated offline augmented images
│   │   ├── images/
│   │   └── labels/
│   └── verified/            # OllamaVerifier confirmed true positive alert photos
│
├── logs/
│   ├── logs.txt             # Alert log: [timestamp] [cam] [event] duration — description | photo: path
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

Four Dahua cameras on the same NVR at `192.168.30.5`:

```json
[
  { "id": 2,  "name": "D2",  "ip": "192.168.30.5", "user": "admin", "pass": "Masters@6677",
    "port": 554, "channel": 2,  "type": "dahua", "subtype": 0, "rtsp_path": "" },
  { "id": 6,  "name": "D6",  "ip": "192.168.30.5", "user": "admin", "pass": "Masters@6677",
    "port": 554, "channel": 6,  "type": "dahua", "subtype": 0, "rtsp_path": "" },
  { "id": 9,  "name": "D9",  "ip": "192.168.30.5", "user": "admin", "pass": "Masters@6677",
    "port": 554, "channel": 9,  "type": "dahua", "subtype": 0, "rtsp_path": "" },
  { "id": 19, "name": "D19", "ip": "192.168.30.5", "user": "admin", "pass": "Masters@6677",
    "port": 554, "channel": 19, "type": "dahua", "subtype": 0, "rtsp_path": "" }
]
```

**RTSP URL construction (Dahua):**  
`rtsp://admin:{url_encoded_pass}@{ip}:{port}/cam/realmonitor?channel={ch}&subtype={subtype}`

**RTSP URL construction (Hikvision):**  
`rtsp://admin:{url_encoded_pass}@{ip}:{port}/Streaming/Channels/{ch}0{subtype+1}`

`subtype=0` = main stream (higher res), `subtype=1` = sub-stream (more concurrent connections).  
The `rtsp_path` field overrides auto-build if non-empty.

**Fallback (no cameras.json):** reads single camera from `.env` variables:
- `CAMERA_IP`, `CAMERA_USER`, `CAMERA_PASS`, `CAMERA_PORT`, `CAMERA_CHANNEL`, `CAMERA_TYPE`, `CAMERA_RTSP_PATH`

---

## 4. Environment Configuration (`.env`)

```env
# Active .env values
CAMERA_IP=192.168.30.5
CAMERA_USER=admin
CAMERA_PASS=Masters@6677
CAMERA_PORT=554
CAMERA_CHANNEL=2
CAMERA_RTSP_PATH=              # empty = auto-build from type
CAMERA_TYPE=dahua

CHECK_INTERVAL_SEC=5
DETECTION_CONF=0.40

SLEEP_THRESHOLD_SEC=120        # 2 min — head-down before SLEEPING alert
TIMEWASTE_THRESHOLD_SEC=600    # 10 min (legacy, not used in v2)
PHONE_COOLDOWN_SEC=30          # legacy, replaced by session logic

CAMERA_ANGLE=overhead          # overhead / side / front
EAR_CONF_VISIBLE=0.55          # keypoint conf >= this → ear visible
EAR_CONF_HIDDEN=0.20           # keypoint conf <= this → ear hidden
STANDING_RATIO=1.3             # bbox h/w > this = standing
LEANING_BACK_RATIO=2.0

USE_OLLAMA=true
OLLAMA_MODEL=llava:latest
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

**Optional (installed manually):**
- `insightface` + `onnxruntime-gpu` — FaceWorker face detection
- `gunicorn` — production web server for web_ui (run.sh uses it if available)
- `pyyaml` — required by train.py for dataset.yaml writing

---

## 6. `monitor.py` — Complete Technical Specification

### 6.1 Module-Level Constants and Config

```python
import os, time, signal, threading, json, urllib.parse, math
from collections import deque
from datetime import datetime
from pathlib import Path
import cv2, torch, numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# GPU / precision
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
HALF        = DEVICE == "cuda"          # fp16 on GPU, fp32 on CPU

# Detection confidence thresholds
CONF_PERSON = 0.50   # YOLO confidence for person detections
CONF_PHONE  = 0.35   # balanced: custom model true positives ~35%+, false ~10-25%
CONF_CHAIR  = 0.40   # chair detection (COCO class 56)
CONF_KP     = 0.50   # keypoint confidence for pose estimation

# Display resolution
DISPLAY_W   = 1280
DISPLAY_H   = 720

# COCO class IDs
PERSON = 0    # COCO class 0
PHONE  = 67   # COCO class 67
CHAIR  = 56   # COCO class 56

# Custom model class IDs (trained, different from COCO)
CUSTOM_PHONE_HAND = 0
CUSTOM_PHONE_EAR  = 1
CUSTOM_PHONE_DESK = 2
CUSTOM_SLEEPING   = 3

# Alert timing (read from .env with defaults)
SLEEP_THRESHOLD      = int(os.getenv("SLEEP_THRESHOLD_SEC",      "120"))  # 2 min
PHONE_SESSION_GRACE  = float(os.getenv("PHONE_SESSION_GRACE_SEC", "6"))   # 6s gap = session ended
PHONE_SESSION_MIN    = float(os.getenv("PHONE_SESSION_MIN_SEC",   "20"))  # ignore sessions < 20s
PHONE_SESSION_MAX    = int(os.getenv("PHONE_SESSION_MAX_SEC",     "600")) # periodic save every 600s

# Motion / stillness
MOTION_STILL_SECS = 60      # seconds still before counts as "motionless"
MOTION_THRESH     = 0.012   # pixel diff ratio < this = still (64×64 grayscale crop)
MIN_TRACK_AGE     = 30      # seconds before track can trigger alert (not currently used)

# Camera angle and ear visibility (from .env)
CAMERA_ANGLE       = os.getenv("CAMERA_ANGLE",       "side").lower()  # overhead in active .env
EAR_CONF_VISIBLE   = float(os.getenv("EAR_CONF_VISIBLE",   "0.50"))
EAR_CONF_HIDDEN    = float(os.getenv("EAR_CONF_HIDDEN",    "0.25"))
STANDING_RATIO     = float(os.getenv("STANDING_RATIO",     "1.6"))
LEANING_BACK_RATIO = float(os.getenv("LEANING_BACK_RATIO", "1.5"))
```

### 6.2 Shared GPU Models

Three models loaded once at startup, shared across all cameras:

```python
_detect_model  = None   # yolov8s.pt  — COCO person(0) + phone(67)
_custom_model  = None   # custom_model/weights/best.pt — phone_hand/ear/desk/sleeping
_pose_model    = None   # yolov8n-pose.pt
_infer_lock    = threading.Lock()  # serializes all GPU inference calls
```

**`_load_models()` function:**
1. Loads `yolov8s.pt` → warm-up with 416×416 dummy frame → `device=DEVICE, imgsz=416, half=HALF`
2. If `custom_model/weights/best.pt` exists → load → warm-up with 320×320 dummy frame
3. Loads `yolov8n-pose.pt` → warm-up with 416×416 dummy frame

### 6.3 `EvidenceAcc` — Temporal Voting Accumulator

```python
class EvidenceAcc:
    def __init__(self, window=8, min_ratio=0.70, min_frames=None):
        self._h = deque(maxlen=window)   # rolling history of 0/1 votes
        self._w = window
        self._r = min_ratio              # fraction of window that must be 1
        self._min = min_frames if min_frames is not None else max(3, window // 2)
        # min_frames: minimum observations before confirmation can happen

    def add(self, v):    # v: bool → 1 or 0
    def reset(self):     # clear history

    @property
    def confirmed(self):
        if len(self._h) < self._min: return False
        return (sum(self._h) / len(self._h)) >= self._r

    @property
    def ratio(self): return sum / len if non-empty else 0.0
```

**Phone accumulator params:** `window=10, min_ratio=0.30, min_frames=3`  
(3 frames out of 10 = 30% → reduces flicker; custom model gives 20-38% conf from overhead)

**Sleep accumulator params:** `window=12, min_ratio=0.75`  
(9 of 12 frames must confirm sleeping before alert)

### 6.4 `TrackState` — Per-Person State

All state for one tracked person:

```python
class TrackState:
    def __init__(self, tid):
        self.tid = tid
        self.bbox = None           # (x1,y1,x2,y2) — current bounding box
        self.centroid = (0, 0)
        self.last_seen = time.time()
        self.first_seen = time.time()

        # Phone state
        self.has_phone = False          # YOLO detected phone in hand or ear
        self.phone_on_ear = False       # PoseWorker: wrist near ear keypoint
        self.phone_type = None          # 'hand' | 'ear' | 'desk' | None
        self.phone_bbox = None          # full-frame bbox of detected phone

        # Pose state
        self.pose_sleeping = False      # PoseWorker: nose below shoulders
        self.face_visible = True        # FaceWorker: InsightFace detected face
        self.head_yaw = 0.0             # degrees, from pose nose-ear geometry

        # Wrist keypoints (from PoseWorker) — used in phone classification
        self.left_wrist  = None         # (x, y, conf) or None
        self.right_wrist = None         # (x, y, conf) or None

        # Motion
        self.motion_score = 1.0
        self.is_still = False
        self.still_since = None         # monotonic time when stillness started

        # Evidence accumulators
        self.ev_phone = EvidenceAcc(window=10, min_ratio=0.30, min_frames=3)
        self.ev_sleep = EvidenceAcc(window=12, min_ratio=0.75)

        # Sleep session tracking
        self.sleep_start = None         # monotonic time when sleep session started
        self.sleep_last_active = None   # last time sleep_ok was True
        self.sleep_session_ann = None   # annotated frame saved at sleep start

        # Phone session tracking (photo saved when session ENDS)
        self.phone_session_start = None  # monotonic time session started
        self.phone_session_ptype = None  # 'hand' or 'ear' for this session
        self.phone_last_active   = None  # last time phone_ok was True
        self.phone_session_ann   = None  # best annotated frame from session
        self.phone_session_saved = 0.0   # last periodic save time
        self.phone_total_sec     = 0.0   # cumulative seconds all sessions

    @property
    def track_age(self): return time.time() - self.first_seen

    @property
    def sleep_raw(self):
        # pose_sleeping REQUIRED (no keypoints = no sleep alert)
        # PLUS at least 1 supporting signal:
        #   - face not visible, OR
        #   - body still for > MOTION_STILL_SECS (60s)
        if not self.pose_sleeping: return False
        supporting = [
            not self.face_visible,
            self.is_still and self.still_since is not None
                and (time.time() - self.still_since) > MOTION_STILL_SECS,
        ]
        return sum(supporting) >= 1

    @property
    def phone_raw(self):
        # Requires YOLO to have actually detected a phone object
        # phone_on_ear alone (wrist near ear but no phone object) = ignored
        # → prevents "scratching head / adjusting glasses" false positives
        yolo_detected  = self.phone_type in ('hand', 'ear')
        pose_confirmed = self.phone_on_ear and self.has_phone
        return yolo_detected or pose_confirmed
```

### 6.5 `CameraState` — Thread-Safe Per-Camera Frame + Detection State

```python
class CameraState:
    def __init__(self):
        self._l = threading.Lock()
        self.frame   = None    # latest raw OpenCV frame (BGR numpy array)
        self.persons = []      # list of dicts: {track_id, bbox, conf, centroid}
        self.phones  = []      # list of dicts: {bbox, conf, track_id, type}
        self.alerts  = []      # list of overlay string messages

    def set_frame(self, f)   # thread-safe frame write
    def get_frame(self)      # thread-safe frame read
    def update(self, **kw)   # thread-safe attribute set
    def snapshot(self)       # returns dict(frame, persons, phones, alerts) copy
```

### 6.6 `SimpleTracker` — Per-Camera IoU Tracker

No ByteTrack — simple greedy IoU matching per camera (avoids multi-camera ID conflicts):

```python
class SimpleTracker:
    def __init__(self):
        self._next = 1       # next track ID to assign
        self._active = {}    # tid → {bbox, age}

    @staticmethod
    def _iou(a, b):          # standard IoU computation

    def update(self, dets):
        # dets: [(bbox, conf), ...]
        # Returns: [(bbox, conf, track_id), ...]
        # Match threshold: IoU > 0.30
        # Stale removal: age >= 45 frames (removes disappeared tracks)
```

**IoU match threshold:** `0.30`  
**Stale track TTL:** `45 frames` (roughly 22 seconds at batch ~0.5s)  
**Track purge from TrackState dict:** `900 seconds` of inactivity

### 6.7 Phone Classification — `_classify_phone()`

```python
def _classify_phone(phone_bbox, person_bbox, phone_on_ear_signal=False,
                    left_wrist=None, right_wrist=None):
    """Returns: 'ear' | 'hand' | 'desk'"""
    
    # Priority 1: PoseWorker ear signal → 'ear'
    if phone_on_ear_signal: return 'ear'
    
    # Priority 2: Phone center in top 28% of person bbox → 'ear'
    rel_y = (phone_cy - ey1) / person_height
    if rel_y < 0.28: return 'ear'
    
    # Priority 3: Wrist keypoint within MARGIN=30px of phone bbox → 'hand'
    MARGIN = 30   # px — must overlap or be within ~1 phone-width
    # min wrist confidence: 0.30
    # check both left_wrist and right_wrist
    if wrist within MARGIN of phone: return 'hand'
    
    # Default: phone not held → 'desk'
    return 'desk'
```

**Key constants:**
- Ear zone top threshold: `0.28` (top 28% of person bbox = head region)
- Wrist margin: `30 px` (previously 60px = too loose, 12px = too tight for overhead angle)
- Wrist confidence minimum: `0.30`

### 6.8 Phone Validity Filter — `_valid_phone()`

```python
def _valid_phone(phone_bbox, person_bbox, frame_h, frame_w):
    """Returns True if the detected phone bbox is plausible."""
    
    aspect = bbox_height / bbox_width
    
    # Shape: portrait (tall) OR landscape (wide)
    portrait  = 1.3 <= aspect <= 4.5    # tightened from v1 (1.2–5.0)
    landscape = 0.22 <= aspect <= 0.77  # tightened from v1 (0.2–0.83)
    if not (portrait or landscape): return False
    
    # Size: relative to frame area
    area_ratio = bbox_area / frame_area
    if area_ratio < 0.0003: return False   # too small (mouse/pen)
    if area_ratio > 0.04:   return False   # too large (monitor/whiteboard)
    
    # Position: phone center must be in 5%–95% of person bbox height
    # v1 used 20%–85% — phones held at ear level were cut off
    if not (ey1 + height*0.05 <= phone_cy <= ey1 + height*0.95): return False
    
    return True
```

### 6.9 `CameraSession` — Config + State Bundle Per Camera

```python
class CameraSession:
    def __init__(self, cfg):
        self.cam_id  = cfg['id']
        self.name    = cfg.get('name', f"Cam-{cfg['id']}")
        self.cfg     = cfg
        self.state   = CameraState()
        self.tracks  = {}             # tid → TrackState
        self.tracks_lock = threading.Lock()
        self.tracker = SimpleTracker()

    def rtsp_url(self):
        # Builds RTSP URL from cfg
        # If cfg['rtsp_path'] is set, appends it directly
        # Dahua: /cam/realmonitor?channel={ch}&subtype={subtype}
        # Hikvision: /Streaming/Channels/{ch}0{subtype+1}
        # Password is URL-encoded with urllib.parse.quote(pass, safe='')
```

### 6.10 `BatchDetectWorker` — Two-Pass GPU Detection Thread

**Thread name:** `"BatchDetect"`  
**Run loop interval:** max(0, 0.5 - elapsed) seconds (targeting 2Hz detection)

**Two-pass algorithm:**

**Pass 1 — Person detection on full frames (all cameras batched together):**
```python
with _infer_lock:
    r1 = _detect_model(frames, verbose=False, conf=CONF_PERSON,
                       imgsz=416, device=DEVICE, half=HALF,
                       classes=[PERSON])  # class 0 only
```
- Input: list of raw BGR frames (one per live camera)
- Output: person bboxes per frame
- Track persons using SimpleTracker
- Update TrackState.bbox, centroid, standing
- Collect person crops with 25px padding for Pass 2

**Pass 2 — Phone detection on batched person CROPS:**
```python
# Custom model (skip _valid_phone — custom bboxes are person-sized)
r2_custom = _custom_model(crops, verbose=False, conf=CONF_PHONE,
                          imgsz=320, device=DEVICE, half=HALF,
                          classes=[CUSTOM_PHONE_HAND, CUSTOM_PHONE_EAR, CUSTOM_PHONE_DESK])

# COCO base model (always runs as fallback, conf lowered to 0.25 for crop)
r2_coco = _detect_model(crops, verbose=False, conf=0.25,
                         imgsz=320, device=DEVICE, half=HALF,
                         classes=[PHONE])  # class 67
```

**Phone merging logic (priority order):**
1. Custom model hand/ear detections added first (`skip_valid=True` — bboxes are person-sized not phone-sized)
2. COCO model fills gaps custom model missed
3. `hand` or `ear` type NEVER downgraded to `desk`
4. Existing hand/ear detection blocks a desk detection for same track_id

**After detection — state cleanup:**
- For each person NOT in detected phones this frame → clear `has_phone`, `phone_type`, `phone_bbox`

**Track purge:** every batch run, remove TrackState entries with `last_seen > 900 seconds` ago

### 6.11 `BatchPoseWorker` — Pose Estimation Thread

**Thread name:** `"BatchPose"`  
**Run loop interval:** max(0, 1.5 - elapsed) seconds (targeting ~0.67Hz)

```python
with _infer_lock:
    results = _pose_model(list(frames), verbose=False, conf=CONF_PERSON,
                          imgsz=416, device=DEVICE, half=HALF)
```

**Keypoint indices (COCO 17-keypoint format):**
- 0: nose
- 3: left ear
- 4: right ear
- 5: left shoulder
- 6: right shoulder
- 9: left wrist
- 10: right wrist

**Sleeping detection:**
```python
avg_sh = mean(shoulder y-coords where conf >= CONF_KP)
sh_span = shoulder x-span (min 40px; 60px if one shoulder; 80px if none)
sleeping = nose_conf >= CONF_KP and avg_sh is not None
           and nose_y > avg_sh + sh_span * 0.4
# nose is 40% of shoulder span BELOW shoulder level = head is down
```

**Phone on ear detection:**
```python
thr_y = sh_span * 0.55   # vertical tolerance
thr_x = sh_span * 0.65   # horizontal tolerance
# Check: |wrist_y - ear_y| < thr_y AND |wrist_x - ear_x| < thr_x
# Both left and right wrist-ear pairs checked
```

**Head yaw estimation:**
```python
# Both ears visible:
ear_mid  = (le_x + re_x) / 2
ear_span = max(|re_x - le_x|, 1)
yaw_deg  = ((nose_x - ear_mid) / ear_span) * 90

# Only left ear visible:  yaw_deg = +55.0 (facing right)
# Only right ear visible: yaw_deg = -55.0 (facing left)
# No ears visible:        yaw_deg =  0.0
```

**Wrist output (for phone classifier):**
- Returned if `wrist_conf >= 0.20` (lower threshold than keypoint general CONF_KP=0.50)

**Track matching:** nose position must fall inside a person bbox from `cam.state.snapshot()['persons']`

### 6.12 `MotionWorker` — Per-Camera Pixel Diff Thread

**Thread name:** `"Motion-{cam_id}"`  
**Run loop interval:** max(0, 0.5 - elapsed) seconds

**Algorithm:**
```python
# For each tracked person:
crop = frame[y1:y2, x1:x2]                      # person bounding box crop
gray = cv2.cvtColor(cv2.resize(crop, (64,64)), cv2.COLOR_BGR2GRAY)
score = cv2.absdiff(gray, prev_gray).mean() / 255.0

is_still = (score < MOTION_THRESH)   # MOTION_THRESH = 0.012
# still_since set when is_still becomes True, cleared when motion resumes
```

### 6.13 `FaceWorker` — Per-Camera InsightFace Thread

**Thread name:** `"Face-{cam_id}"`  
**Run loop interval:** max(0, 2.0 - elapsed) seconds (slowest worker — InsightFace is heavy)

**Model:** `insightface.app.FaceAnalysis(name="buffalo_sc")`  
**Providers:** `["CUDAExecutionProvider", "CPUExecutionProvider"]`  
**Modules:** `["detection"]` only (no recognition/landmarks)  
**det_size:** `(160, 160)` — small because crops are already resized to 160×160

**Logic:**
- For each person crop → resize to 160×160 → run InsightFace detection
- `face_visible = len(faces) > 0`
- If InsightFace unavailable: silently skips, `face_visible` stays `True` (conservative)

### 6.14 `OllamaVerifier` — False-Positive Removal Thread

**Thread name:** `"OllamaVerifier"`  
**Activated by:** `USE_OLLAMA=true` in `.env`  
**Scan interval:** 8 seconds  
**Model:** `os.getenv("OLLAMA_MODEL", "llava:latest")`  
**Endpoint:** `http://localhost:11434/api/generate`

**Scans:** `logs/photos/Cam*.jpg` — up to 15 most recent by mtime

**Questions per event type:**

```
PHONE_HAND question:
  "This is a CEILING/OVERHEAD security camera looking DOWN at an office.
   Look at the person inside the RED or ORANGE bounding box.
   Is that specific person CLEARLY holding a rectangular mobile phone
   in their hand? Look for a small rectangular device in their hand.
   Ignore people in GREEN boxes.
   Answer YES only if you can clearly see a phone in the highlighted person's hand.
   Answer NO if it is a mouse, keyboard, pen, or if no phone is visible.
   Reply with only YES or NO."

PHONE_EAR question:
  "This is a CEILING/OVERHEAD security camera looking DOWN at an office.
   Look at the person inside the RED or ORANGE bounding box.
   Is that specific person CLEARLY holding a mobile phone to their ear
   while talking? From overhead, this looks like a hand raised to the side
   of the head with a small rectangle visible.
   Do NOT answer YES if the person is just touching their face,
   scratching their head, or adjusting glasses.
   Answer YES only if a phone is clearly visible near their ear.
   Reply with only YES or NO."

SLEEPING question:
  "This is a CEILING/OVERHEAD security camera looking DOWN at an office.
   Look at the person inside the RED bounding box.
   Is that person's head resting DOWN on the desk or their arms,
   appearing to be asleep? From overhead, a sleeping person's head
   will be very close to the desk surface.
   Reply with only YES or NO."
```

**Decision logic:**
- `answer="NO"` → delete photo from disk + remove matching line from `logs/logs.txt`
- `answer="YES"` → copy to `training_data/verified/` for future training
- `answer` parsing: only first word matters. Anything uncertain → keep (default YES on error)
- Ollama call params: `temperature=0.05, num_predict=6` (very deterministic, short answer)

### 6.15 `BehaviorEngine` — Per-Camera Alert State Machine Thread

**Thread name:** `"Behavior-{cam_id}"`  
**Run loop interval:** max(0, 0.5 - elapsed) seconds

**`_tick(frame)` — main decision logic:**

```python
# For each track in current camera:
ts.ev_phone.add(ts.phone_raw)    # add current phone signal
ts.ev_sleep.add(ts.sleep_raw)    # add current sleep signal
phone_ok = ts.ev_phone.confirmed
sleep_ok = ts.ev_sleep.confirmed

# CRITICAL: Only alert if person CURRENTLY VISIBLE in this frame
live_bbox = live_bbox_by_tid.get(tid)
if live_bbox is None: continue   # person not in current frame → skip
ts._live_bbox = live_bbox        # used by _force_red_box()
```

**Phone session state machine:**
```
phone_ok=True:
  if phone_session_start is None:
    → NEW SESSION: save IMMEDIATE photo (0s), set session_start, session_saved
  else:
    → ONGOING SESSION: update best frame every 8 seconds
    
  label shown: "PHONE ON EAR" or "PHONE IN HAND"
  overlay: "[{cam.name}#{tid}] {label}  {m}m{s:02d}s"
  
  Periodic save: if session_dur >= PHONE_SESSION_MAX and
                    now - session_saved >= PHONE_SESSION_MAX
    → fire additional alert with session duration

phone_ok=False:
  if phone_session_start is not None and
     phone_last_active is not None and
     now - phone_last_active >= PHONE_SESSION_GRACE (6s):
    → SESSION ENDED:
      duration = phone_last_active - phone_session_start
      phone_total_sec += duration
      if duration >= PHONE_SESSION_MIN (20s):
        → fire SESSION END alert with duration
      reset session state
```

**Sleep session state machine:**
```
sleep_ok=True:
  update sleep_last_active = now
  if sleep_start is None: sleep_start = now
  elapsed = now - sleep_start
  
  if elapsed >= SLEEP_THRESHOLD (120s) and sleep_session_ann is None:
    → PHOTO 1: immediate alert with elapsed duration
    → set sleep_session_ann
  
  overlay: "[{cam.name}#{tid}] SLEEPING {m}m{s:02d}s ({pct}%)"

sleep_ok=False:
  if sleep_start is not None and sleep_last_active is not None
     and sleep_session_ann is not None:
    duration = sleep_last_active - sleep_start
    if duration >= SLEEP_THRESHOLD:
      → PHOTO 2: final alert when they wake up
  reset: sleep_start, sleep_last_active, sleep_session_ann = None
```

### 6.16 `_force_red_box()` — Guaranteed Violator Box

```python
def _force_red_box(ann_frame, ts, ptype, orig_frame):
    """
    Forces a red box on the violating person in the saved alert photo.
    Uses ts._live_bbox (current detection position) not ts.bbox (may be stale).
    Scales bbox if frame was resized (ann_frame vs orig_frame different shapes).
    Draws: RED rectangle + label text ("PHONE ON EAR" / "SLEEPING" / "PHONE IN HAND")
    Label background filled with RED, white text on top.
    """
```

### 6.17 Camera Reader Thread — `camera_reader(cam, startup_delay=0)`

**Startup delay:** cameras staggered by 1.5s each to avoid simultaneous NVR connection burst
(0s, 1.5s, 3.0s, 4.5s for 4 cameras)

**RTSP connection:**
```python
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimum buffer → always latest frame
```

**Reconnect logic:**
- On open failure: retry_delay starts at 3s, increments +2 each attempt, max 15s
- On read failure ("stream lost"): reconnect after 3s

**Frame sharing with web_ui:**
```python
# Every 3rd frame (~3fps write rate)
# Shared frames directory: /tmp/monitor_frames/
# Write to .tmp.jpg first, then atomic rename to .jpg (avoids partial reads)
shared = cv2.resize(frame, (640, 360))
tmp = SHARED_FRAMES_DIR / f"{cam.name}.tmp.jpg"
cv2.imwrite(str(tmp), shared, [cv2.IMWRITE_JPEG_QUALITY, 72])
tmp.rename(SHARED_FRAMES_DIR / f"{cam.name}.jpg")
```

### 6.18 Annotation — `annotate_cam(cam, target_w=None, target_h=None)`

**Person box colors:**
- Green `(0,200,0)` = OK / working normally
- Red `(0,0,220)` = PHONE IN HAND (ev_phone confirmed)
- Deep orange-red `(0,60,255)` = PHONE ON EAR
- Red `(0,0,220)` = SLEEPING

**Phone object box colors:**
- `(0,60,255)` = PHONE ON EAR
- `(100,100,100)` = PHONE ON DESK (grey — not a violation)
- `(0,140,255)` = PHONE IN HAND (orange)

**Per-person info text:**
- Track ID + cumulative phone time: `"#{tid}  Phone: {m}m{s:02d}s"` (only if total_phone > 5s)
- Font: HERSHEY_SIMPLEX 0.38, gray `(180,180,180)`

**Status bar:** bottom 26px, dark background `(30,30,30)`, shows:  
`"{cam.name} | HH:MM:SS | Persons:{n}  Phone:{n}  Sleeping:{n}"`

**Grid layout for multiple cameras:**
- Max 4 columns, rows = ceil(n/4)
- Each cell = DISPLAY_W//cols × DISPLAY_H//rows

### 6.19 `main()` Startup Sequence

1. Print GPU info (device name, VRAM)
2. `_load_models()` — load all three models
3. `_load_camera_configs()` — from cameras.json or .env
4. Create `CameraSession` for each camera
5. Start RTSP reader threads — staggered by 1.5s each
6. Wait up to 15 seconds for at least one live frame (poll 100ms × 150 times)
7. Start `BatchDetectWorker(cameras)` (single shared thread)
8. Start `BatchPoseWorker(cameras)` (single shared thread)
9. For each camera: start `MotionWorker`, `FaceWorker`, `BehaviorEngine`
10. Start `OllamaVerifier()` (single global thread)
11. Headless mode (`HEADLESS=true` default): `while _running: time.sleep(1)`
12. Developer mode (`HEADLESS=false`): `cv2.imshow` with `make_grid` or single-cam view

**Signal handling:** `SIGINT` and `SIGTERM` both set `_running = False`

---

## 7. `rules.py` — Alert Handler

### 7.1 Constants

```python
USE_OLLAMA   = os.getenv("USE_OLLAMA",   "false").lower() == "true"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava:latest")

LOGS_DIR   = "logs"
PHOTOS_DIR = "logs/photos"
LOGS_FILE  = "logs/logs.txt"

_BANNER = {
    "PHONE_HAND": (20,  90, 200),    # dark blue
    "PHONE_EAR":  (0,   60, 220),    # dark blue-red
    "SLEEPING":   (40,  40, 160),    # dark purple-blue
}
_LABEL = {
    "PHONE_HAND": "PHONE IN HAND",
    "PHONE_EAR":  "PHONE ON EAR / CALLING",
    "SLEEPING":   "SLEEPING / HEAD ON DESK",
}
_DEFAULT_MSG = {
    "PHONE_HAND": "Employee detected holding phone in hand.",
    "PHONE_EAR":  "Employee detected talking on phone / phone held to ear.",
    "SLEEPING":   "Employee appears sleeping — head down, no face visible, or motionless.",
}
_OLLAMA_PROMPT = {
    "PHONE_HAND": "In one sentence, describe what the employee is doing with the phone in their hand.",
    "PHONE_EAR":  "In one sentence, describe the employee talking on or holding a phone to their ear.",
    "SLEEPING":   "In one sentence, describe the employee's posture suggesting they are sleeping.",
}
```

### 7.2 `fire_alert(event, frame, det, elapsed_sec, cam_id, cam_name)`

```python
# 1. Generate filename: Cam{cam_id}_{event}_{YYYYMMDD_HHMMSS}.jpg
# 2. Call _draw() to add event banner to frame
# 3. cv2.imwrite(photo_path, annotated_frame)
# 4. Write log line to logs/logs.txt (append):
#    "[{timestamp}]  [Cam{cam_id}:{cam_name}]  [{event}]  {m}m {s:02d}s  —  {msg}  |  photo: {path}\n"
# 5. Print to stdout with red ANSI color: \033[91m[ALERT]\033[0m
# 6. If USE_OLLAMA=true: spawn background thread → _ollama_update()
```

### 7.3 `_draw(frame, event, det, elapsed_sec, cam_name)`

```python
# 1. Resize to 1280×720 if not already
# 2. Top banner (full-width, 80px height):
#    - Background color from _BANNER dict
#    - Text: "  [{cam_name}]  {_LABEL[event]}   |   Total: {dur_str}"
#    - Font scale 1.4, auto-reduced if text too wide
#    - White text, thickness 3, cv2.LINE_AA
# 3. Timestamp bottom-right:
#    "YYYY-MM-DD  HH:MM:SS", scale 0.6, white (220,220,220)
# Frame already has colored bboxes from annotate_cam() — only banner is added
```

### 7.4 Ollama Integration

`_ollama_update(photo, event, old_line)`:
- Calls `_ollama_describe(photo, event)` in background thread
- If description returned: replaces `_DEFAULT_MSG[event]` with actual description in logs.txt

`_ollama_describe(image_path, event)`:
- POST to `http://localhost:11434/api/generate`
- `{"model": OLLAMA_MODEL, "prompt": _OLLAMA_PROMPT[event], "images": [base64], "stream": False}`
- Timeout: 20 seconds
- Returns `.json().get("response","").strip()` or `""` on error

---

## 8. `web_ui.py` — Flask Web Dashboard

### 8.1 Architecture

**Critical design decision:** web_ui.py does NOT open any RTSP connections. Instead:
- `monitor.py` writes frames to `/tmp/monitor_frames/{cam_name}.jpg` (640×360, JPEG quality 72)
- `web_ui.py` reads those files at 150ms polling interval
- This uses ZERO additional NVR connections beyond the 4 already held by monitor.py

**Flask app:** `app = Flask(__name__)`  
**Bind:** `0.0.0.0:5000`  
**Production server:** gunicorn with `--workers=1 --threads=50 --worker-class=gthread`

### 8.2 API Response Cache

```python
_api_cache  = {}           # key → (data, timestamp)
_cache_lock = threading.Lock()

def _cached(key, ttl_sec, producer):
    """Return cached result if fresh, else call producer() and cache."""

def _bust(key):
    """Invalidate a cache entry immediately."""
```

Cache TTLs:
- `alerts` API: 4.0 seconds
- `cam_alerts_{name}` API: 4.0 seconds
- `logs` API: 5.0 seconds

### 8.3 `CameraStream` — Frame Reader

```python
class CameraStream:
    def __init__(self, cam):
        self._file = SHARED_FRAMES_DIR / f"{self.name}.jpg"
        self._lock = threading.Lock()
        self._frame = None
        threading.Thread(target=self._poll, daemon=True).start()

    def _poll(self):
        # Checks file mtime every 150ms
        # If mtime changed: reads file → np.frombuffer → cv2.imdecode
        # Sets self.connected = True when frame loaded

    def jpeg(self, w=640, h=360, q=70):
        # Returns JPEG bytes, resized to w×h
        # Returns grey placeholder with camera name if no frame
```

### 8.4 Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Main dashboard HTML (inlines camera names as JS constants) |
| `/annotate` | GET | Training annotation page |
| `/snapshot/<name>` | GET | 640×360 JPEG snapshot from shared frame file |
| `/snapshot/<name>/hd` | GET | 1280×720 JPEG snapshot (higher quality for annotation) |
| `/photos/<filename>` | GET | Serve alert photo from logs/photos/ |
| `/api/status` | GET | Camera connected status dict |
| `/api/alerts` | GET | Recent alert sessions grouped by cam+event (TTL 4s) |
| `/api/camera/<name>/alerts` | GET | Alerts for specific camera only |
| `/api/logs` | GET | Last 60 log entries (TTL 5s) |
| `/api/annotate` | POST | Save annotation: cam, label, box → training_data/ |
| `/api/annotation_counts` | GET | Per-class annotation counts + total image count |
| `/api/train` | POST | Start train.py subprocess (needs ≥15 images) |
| `/api/train_status` | GET | Training status: idle/running/done/error + progress line |

### 8.5 Alert Parsing — `_parse_photo(path)` and `_sessions(photos, gap_min=4)`

**Photo filename format:** `Cam{N}_{EVENT}_{YYYYMMDD}_{HHMMSS}.jpg`

`_parse_photo()` extracts:
- `cam`: "Cam2", "Cam6", etc.
- `event`: "PHONE HAND", "SLEEPING", etc. (parts between cam and date)
- `date`, `time` (formatted HH:MM:SS)
- `mtime`: parsed from filename (not filesystem) — accurate even after file copies

`_sessions()` groups consecutive same-cam+event photos within `gap_min=4` minutes into sessions:
- Session has: `first_time`, `last_time`, `first_url`, `last_url`, `photos[]`
- `duration_str`: "X min YY sec" or "Y sec"
- `duration_label`: "{X}m {YY}s" or "{Y}s"
- Sorted newest-first by `last_mtime`

### 8.6 Training Data Annotation

**Classes (index → name):**
```python
CLASSES = ["phone_hand", "phone_ear", "phone_desk", "sleeping", "working"]
CLASS_IDS = {c: i for i, c in enumerate(CLASSES)}
# 0: phone_hand, 1: phone_ear, 2: phone_desk, 3: sleeping, 4: working
```

**`/api/annotate` POST payload:**
```json
{"cam": "D2", "label": "phone_hand", "box": {"x1": 0.3, "y1": 0.1, "x2": 0.7, "y2": 0.9}}
```

**Saved files:**
- Image: `training_data/images/{cam}_{label}_{YYYYMMDD_HHMMSS_ffffff}.jpg` (1280×720 HD snapshot)
- Label: `training_data/labels/{same_stem}.txt` (YOLO format: `{class_id} {cx} {cy} {w} {h}`)
- "working" label → empty `.txt` file (negative sample, no objects)

**Training trigger:**
- Minimum 15 images required
- Spawns `subprocess.Popen(["python3", "train.py"])`
- Stdout/stderr piped to `logs/train.log`
- Status polled via `/api/train_status` every 5 seconds

### 8.7 Dashboard HTML (inline in `web_ui.py`)

**Single-page app, dark theme:**

Color palette:
- Background: `#0b0b0b`
- Cards: `#141414`, borders `#1e1e1e`
- Green: `#2ecc71`, Red: `#e74c3c`, Purple: `#9b59b6`
- Blue accent: `#4a90d9`

**Layout:**
```
┌─ Header (50px) ─────────────────────────────────────────────────────────┐
│  🔴 Employee Monitor  [Live N] [Alerts N] [📚 Training] [HH:MM:SS]      │
└──────────────────────────────────────────────────────────────────────────┘
┌─ Camera Panel (flex:1) ─────────────────┬─ Side Panel (310px) ──────────┐
│  4-column grid of camera cards          │  Recent Alerts (58% height)   │
│  Each card:                             │    - thumbnail + event + dur  │
│    ┌─ cam name + LED indicator ─────┐   ├── Activity Log (42% height)  │
│    │  img (16:9, snapshot poll)     │   │    - log rows with thumb      │
│    └─ footer (alert status) ────────┘   └───────────────────────────────┘
└─────────────────────────────────────────┘
```

**Snapshot polling (replaces MJPEG):**
```javascript
const POLL_MS = 600     // 600ms per camera (about 1.67fps display)
const STAGGER = 85      // ms between cameras to avoid simultaneous requests

// First load: immediate, staggered by 85ms per camera
// Continuous loop via setInterval with staggered setTimeout calls
// Each request: new Image() → onload: set img.src (flicker-free)
```

**Camera modal:** click camera → enlarged HD snapshot (1280×720, 400ms poll = 2.5fps) + camera-specific alerts panel

**Lightbox:** click alert photo → full-screen view with event + times + duration

**Alert card colors:**
- Phone events: red `#e74c3c`
- Sleep events: purple `#9b59b6`
- Duration: green `#2ecc71`

**Camera card states:**
- `.cam-card` normal → border `#1e1e1e`
- `.cam-card.alert` → border `#c0392b` + red glow shadow

### 8.8 Annotation UI HTML (inline in `web_ui.py`)

**Step-by-step workflow:**
1. Select camera (dropdown)
2. Click ❄ Freeze when behavior occurs
3. Draw bounding box on canvas (drag mouse)
4. Select label (5 buttons: Phone Hand, Phone Ear, Phone Desk, Sleeping, Working)
5. Click 💾 Save Annotation
6. Auto-unfreezes for next annotation
7. When ≥15 total images: 🚀 Train Model button enabled

**Canvas drawing:**
- HTML5 Canvas overlaid on top of snapshot image
- Normalized coordinates (0-1) for label storage
- Color: red for phone labels, purple for sleeping, green for working
- Corner handles drawn as white squares

**Model badge:** shows "Base Model" (blue) or "Custom Model" (green) based on `has_model` from `/api/train_status`

---

## 9. `train.py` — Custom Model Training Pipeline

### 9.1 Constants

```python
TRAIN_DIR  = Path("training_data")
AUG_DIR    = TRAIN_DIR / "augmented"
OUT_DIR    = Path("custom_model")
SPLIT_FILE = TRAIN_DIR / "split.json"
CLASSES    = ["phone_hand", "phone_ear", "phone_desk", "sleeping", "working"]
MIN_IMAGES = 15          # minimum required to start training
TARGET_PER_CLASS = 60    # target annotations per class via augmentation
```

### 9.2 Image Quality Filter — `is_good_quality(img_path)`

```python
def blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()    # higher = sharper

def brightness_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.mean()

# Thresholds:
min_blur   = 50    # Laplacian variance minimum
min_bright = 20    # mean pixel value minimum
max_bright = 235   # mean pixel value maximum
```

### 9.3 Preprocessing — `preprocess_image(img)`

CLAHE on L channel in LAB color space:
```python
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b_ch = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l = clahe.apply(l)
lab = cv2.merge([l, a, b_ch])
return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
```

Applied to every original image before augmentation. Handles mixed office lighting (windows + fluorescent).

### 9.4 Offline Augmentation — 10 Operations

```python
AUGMENT_OPS = [
    "hflip",         # cv2.flip(img, 1) — labels: cx = 1 - cx
    "bright_up",     # +45 pixel value (clipped to 255)
    "bright_down",   # -40 pixel value (clipped to 0)
    "contrast_up",   # alpha=1.35, beta=-20
    "contrast_down", # alpha=0.70, beta=30
    "noise",         # Gaussian noise mean=0, std=12
    "blur_slight",   # horizontal motion blur (5×5 kernel, row 2 = 1/5 each)
    "rot90",         # cv2.ROTATE_90_CLOCKWISE — labels rotated accordingly
    "hflip_bright",  # flip + +30 brightness
    "clahe_boost",   # strong CLAHE clipLimit=4.0 (vs 2.0 in preprocessing)
]
```

**Label transforms for geometric ops:**

Horizontal flip: `cx_new = 1.0 - cx` (cy, w, h unchanged)

90° clockwise rotation:
```python
# Input: (cx, cy, bw, bh) in old frame (h×w)
# Output: (1.0-cy, cx, bh, bw) in new frame (w×h)
```

Brightness/contrast/noise/blur: no label transform needed (boxes unchanged)

**Augmentation priority:** images containing rare classes get more augmentations:
```python
deficit = max(TARGET_PER_CLASS - class_counts[c] for c in img_classes)
n_aug   = min(len(AUGMENT_OPS), max(2, deficit // max(len(img_classes), 1)))
ops = random.sample(AUGMENT_OPS, min(n_aug, len(AUGMENT_OPS)))
```

Augmented images saved as JPEG quality 92.

### 9.5 Fixed Train/Val Split — `get_or_create_split(val_ratio=0.18)`

- First run: shuffle originals, take `max(2, floor(n * 0.18))` as validation set
- Saved to `training_data/split.json` — never changes for existing images
- New images added since last split → appended to train (never val)
- Augmented images: ALWAYS train only, never val
- Ensures val images are truly unseen, no augmentation leakage

### 9.6 Auto Model Selection — `pick_base_model(n_train)`

```python
# If custom_model/weights/best.pt exists → continue training from it
# If n_train >= 200 → yolov8m.pt (medium model)
# Else → yolov8s.pt (small model, safer for small datasets with frozen backbone)
```

### 9.7 Training Hyperparameters — `model.train()`

```python
model.train(
    data    = str(yaml_path),           # training_data/dataset.yaml
    epochs  = 200 if n_train < 200 else 150,
    imgsz   = 640,                      # higher res than inference (320 for crops)
    batch   = 8 if n_train < 200 else 16,
    device  = "cuda" if cuda else "cpu",

    # Learning rate schedule
    lr0     = 0.0003,                   # initial learning rate
    lrf     = 0.003,                    # final LR = lr0 * lrf
    cos_lr  = True,                     # cosine annealing
    warmup_epochs   = 5,
    warmup_momentum = 0.8,
    momentum        = 0.937,

    # Regularization (anti-overfitting for small datasets)
    weight_decay    = 0.001,
    label_smoothing = 0.1,
    dropout         = 0.1,

    # Frozen backbone — only detection heads are trained
    freeze  = 15 if n_train < 150 else 10,  # freeze first N layers

    # Online augmentation (in addition to offline)
    mosaic      = 1.0,          # mosaic augmentation probability
    copy_paste  = 0.2,
    mixup       = 0.08,
    degrees     = 12.0,         # rotation ±12°
    translate   = 0.12,         # translation ±12%
    scale       = 0.55,         # scale ±55%
    shear       = 2.0,
    perspective = 0.0003,
    fliplr      = 0.5,          # horizontal flip 50%
    flipud      = 0.0,          # no vertical flip (overhead cameras have fixed orientation)
    hsv_h       = 0.02,
    hsv_s       = 0.7,
    hsv_v       = 0.4,
    erasing     = 0.35,
    close_mosaic= 30,           # disable mosaic last 30 epochs

    # Training control
    patience    = 40,           # early stop if no improvement for 40 epochs
    project     = str(OUT_DIR), # "custom_model"
    name        = "weights",    # saves to custom_model/weights/
    exist_ok    = True,
    verbose     = True,
    plots       = True,
    save_period = 10,           # save checkpoint every 10 epochs
)
```

**Output model:** copies best.pt to `custom_model/weights/best.pt`

### 9.8 Validation Reporting

After training, runs `model.val(split="val", imgsz=640)` and prints:
- Overall mAP@50 and mAP@50-95
- Per-class AP@50 with ✓ (≥75%), ⚠ (≥50%), ✗ (<50%) symbols
- Overfitting risk: HIGH if mAP@50 < 0.65

### 9.9 CLI Interface

```bash
python3 train.py              # augment + train (default)
python3 train.py --check      # show dataset stats only
python3 train.py --augment    # run augmentation only
python3 train.py --eval       # evaluate existing model on val set
python3 train.py --no-augment # skip augmentation, train on existing data
```

---

## 10. `rules.py` — Detailed Alert File Writing

### Log file format

Each line appended to `logs/logs.txt`:
```
[YYYY-MM-DD HH:MM:SS]  [Cam{N}:{cam_name}]  [{EVENT}]  {M}m {S:02d}s  —  {description}  |  photo: {path}
```

If Ollama is enabled, the `{description}` is initially `_DEFAULT_MSG[event]` and then replaced async with Ollama's actual description.

### Photo filename format

`logs/photos/Cam{cam_id}_{EVENT}_{YYYYMMDD_HHMMSS}.jpg`

Example: `Cam2_PHONE_HAND_20260602_143149.jpg`

---

## 11. `run.sh` — One-Command Launcher

**Startup sequence:**
1. `cd` to script directory (works from any CWD)
2. **Clean:** kill old instances by PID files + by path matching (project-specific only, does NOT kill attendance system)
   - `pkill -9 -f "${PROJ}/monitor.py"`
   - `pkill -9 -f "${PROJ}/web_ui.py"`
   - `fuser -k 5000/tcp`
   - Waits up to 8 seconds for clean exit
3. **Check Python:** `python3 --version`
4. **Install deps:** `pip install -q -r requirements.txt`
5. **Environment:**
   ```bash
   export QT_QPA_PLATFORM=xcb
   export QT_LOGGING_RULES="*.warning=false"
   export OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp|loglevel;error"
   export HEADLESS=true
   ```
6. **Clear logs:** truncate `logs/monitor.log` and `logs/web_ui.log`
7. **Start monitor.py:** `python3 -u monitor.py > logs/monitor.log 2>&1 &`; sleep 3; verify PID alive
8. **Start web UI:**
   - If gunicorn available: `gunicorn --workers=1 --threads=50 --worker-class=gthread --bind=0.0.0.0:5000 --timeout=30 --keep-alive=2 --log-file=logs/web_ui.log --log-level=warning web_ui:app`
   - Else: `python3 -u web_ui.py > logs/web_ui.log 2>&1 &`
9. **Print banner** with local IP:5000 and :5000/annotate URLs
10. **Live tail** of `logs/monitor.log` with noise filter (hides InsightFace/ONNX warnings):
    ```bash
    grep -v "Applied providers\|find model\|model ignore\|set det-size\
    \|UserWarning\|CUDAExecutionProvider\|warn(\|onnxruntime\
    \|XDG_SESSION_TYPE\|QT_QPA_PLATFORM\|Ignoring XDG"
    ```
11. **Ctrl+C cleanup:** `trap cleanup SIGINT SIGTERM` — kills both processes, frees port

**PID files:** `/tmp/emp_monitor.pid`, `/tmp/emp_webui.pid`

---

## 12. `record_training.py` — Training Video Recorder

**Purpose:** Record labelled clips directly from live cameras for training data collection.

**Usage:**
```bash
python3 record_training.py --camera D2 --scenario phone_hand
```

**Valid scenarios:** `phone_hand`, `phone_ear`, `phone_desk`, `sleeping`, `working`

**Recording guidance (per scenario):**
```python
REC_GUIDE = {
    "phone_hand": "2–3 min  | show phone in both hands, different positions",
    "phone_ear":  "1–2 min  | left ear, right ear, head turning naturally",
    "phone_desk": "1–2 min  | phone flat on desk, person working normally",
    "sleeping":   "2–3 min  | stay still, different head positions",
    "working":    "3–4 min  | typing, mouse, writing — NO phone visible",
}
```

**Output structure:** `training_videos/{cam_name}/{scenario}/{scenario}_{N:02d}.mp4`

**Controls:** SPACE = start/stop recording, Q/ESC = quit  
**Output codec:** mp4v (MPEG-4), max 1280×720, max 25fps  
**Recommended clips:** 5 per scenario (3 for "working")

---

## 13. `preview_cameras.py` — Camera Grid Preview

**Purpose:** See all cameras in a grid before planning training video recording.

**Output:** OpenCV window showing all cameras in a 4-column grid (640×380 per cell)  
**Controls:** Q = quit  
**Status indicator:** green "Live" / grey error text per camera  
**RTSP:** subtype=0 (main stream) — same URL building as record_training.py

---

## 14. Detection + Alert Pipeline — End-to-End Flow

```
[Camera RTSP] → camera_reader thread
                 ↓ (every frame)
              cam.state.set_frame(frame)
                 ↓ (every 3rd frame)
              /tmp/monitor_frames/{name}.jpg  → web_ui reads this

[BatchDetectWorker — every 0.5s]
  Pass 1: yolov8s(all_frames, imgsz=416, classes=[PERSON])
    → person bboxes → SimpleTracker → TrackState updates
    → person crops collected (25px padding)
  Pass 2a: custom_model(crops, imgsz=320, classes=[0,1,2]) if exists
    → phone_hand/ear/desk detections → added to phones list (skip_valid=True)
  Pass 2b: yolov8s(crops, imgsz=320, classes=[PHONE], conf=0.25)
    → COCO phone detections → _classify_phone() → _valid_phone() filter
    → merged into phones list (hand/ear never downgraded to desk)
  → cam.state.update(persons=..., phones=...)

[BatchPoseWorker — every 1.5s]
  yolov8n-pose(all_frames, imgsz=416)
    → keypoints: sleeping, phone_on_ear, yaw_deg, left_wrist, right_wrist
    → matched to TrackState by nose position inside person bbox
  → ts.pose_sleeping, ts.phone_on_ear, ts.head_yaw, ts.left/right_wrist

[MotionWorker — every 0.5s, per camera]
  64×64 grayscale pixel diff on person crop
    → ts.motion_score, ts.is_still, ts.still_since

[FaceWorker — every 2.0s, per camera]
  InsightFace(160×160 person crop)
    → ts.face_visible

[BehaviorEngine — every 0.5s, per camera]
  For each active track:
    ts.ev_phone.add(ts.phone_raw)   # phone_raw = YOLO detected OR pose confirmed
    ts.ev_sleep.add(ts.sleep_raw)   # sleep_raw = pose_sleeping + 1 supporting signal
    
    phone_ok = ev_phone.confirmed   # 3 of 10 frames (30%) = confirmed
    sleep_ok = ev_sleep.confirmed   # 9 of 12 frames (75%) = confirmed
    
    ONLY if person currently visible (live_bbox_by_tid check):
      → Phone session state machine (immediate + session-end photos)
      → Sleep session state machine (threshold + wake-up photos)
      → build overlay text list
  
  → cam.state.update(alerts=overlays)
  → rules.fire_alert() calls

[OllamaVerifier — every 8s]
  Scans logs/photos/ for new Cam*.jpg files
  → POST to Ollama llava with event-specific YES/NO question
  → NO: delete photo + remove from logs.txt
  → YES: copy to training_data/verified/
```

---

## 15. Thread Summary

| Thread Name | Count | Interval | Purpose |
|-------------|-------|----------|---------|
| `camera_reader` | 1 per camera | continuous | RTSP frame grab + shared file write |
| `BatchDetect` | 1 (shared) | 0.5s | Person + phone detection on all cameras |
| `BatchPose` | 1 (shared) | 1.5s | Pose keypoints on all cameras |
| `Motion-{id}` | 1 per camera | 0.5s | Pixel diff stillness detection |
| `Face-{id}` | 1 per camera | 2.0s | InsightFace face visibility |
| `Behavior-{id}` | 1 per camera | 0.5s | Alert state machine |
| `OllamaVerifier` | 1 (global) | 8s | LLaVA false-positive removal |
| `CameraStream._poll` | 1 per camera (web_ui) | 0.15s | Read shared frame files |

For 4 cameras: **14 threads total** (monitor.py) + **4 threads** (web_ui.py CameraStream polls)

---

## 16. Model Details

### 16.1 `yolov8s.pt` — Primary Detection Model

- **Architecture:** YOLOv8 Small (COCO pre-trained)
- **Used for:** Person detection (class 0) in Pass 1, Phone detection (class 67) in Pass 2
- **Inference params:** `imgsz=416, half=True (fp16 on GPU), conf=0.50 (persons) / 0.25 (phones on crops)`
- **CANNOT be replaced by custom model** — uses COCO class IDs (0=person, 67=phone)

### 16.2 `yolov8n-pose.pt` — Pose Estimation Model

- **Architecture:** YOLOv8 Nano Pose (COCO pre-trained)
- **Used for:** 17-keypoint skeleton detection (sleeping, phone-on-ear, head yaw, wrist positions)
- **Inference params:** `imgsz=416, half=True, conf=0.50`

### 16.3 `custom_model/weights/best.pt` — Custom Trained Model

- **Architecture:** YOLOv8 Small or Medium, fine-tuned from COCO checkpoint
- **Classes:** 0=phone_hand, 1=phone_ear, 2=phone_desk, 3=sleeping, 4=working
- **Training data:** 85 images (overhead camera angle, this specific office)
- **Inference params:** `imgsz=320, half=True, conf=0.35`
- **Note:** bboxes are person-sized (annotated around person, not phone) → `skip_valid=True`
- **Activation:** if `custom_model/weights/best.pt` exists, it runs alongside base model (both results merged)

### 16.4 `yolo26n.pt` — Not Used

Available in project root but not loaded by monitor.py v2.

---

## 17. Key Design Decisions and Why

### 17.1 Two-pass detection (person first, then phone on crop)
**Why:** Phones are small relative to full-frame resolution. Cropping to person and re-running at 320px gives ~4× more pixels per phone, significantly better recall than single-pass full-frame detection.

### 17.2 Batch inference (all cameras in one GPU call)
**Why:** GPU has high per-call overhead. Batching 4 cameras reduces per-camera inference time from ~50ms to ~15ms. Single GPU call instead of 4 separate calls.

### 17.3 fp16 half-precision
**Why:** ~2× memory bandwidth reduction, ~2× speedup on modern NVIDIA GPUs with no meaningful accuracy loss for detection at this confidence threshold.

### 17.4 SimpleTracker (not ByteTrack)
**Why:** ByteTrack has complex state that can cause ID conflicts when multiple cameras share the same tracker instance. SimpleTracker is per-camera, stateless across cameras, uses simple IoU matching — sufficient for one person per desk office monitoring.

### 17.5 Session-based phone timing (photo at session START + session END)
**Why:** Two photos per event gives better evidence:
1. Start photo proves it's happening (live detection)
2. End photo proves duration (how long they used phone)
The PHONE_SESSION_GRACE (6s) gap before declaring session ended prevents one phone detection → one alert per frame.

### 17.6 Custom model runs ALONGSIDE base model (not replacing it)
**Why:** Custom model trained on overhead angle is better at classifying hand/ear/desk but has inconsistent detection confidence (20-38%). Base COCO model catches phones the custom model misses. Combining gives best recall.

### 17.7 Shared frame files (/tmp/monitor_frames/) instead of MJPEG
**Why:** MJPEG streams hold a gunicorn thread open permanently (1 per camera = 4 blocked threads). Snapshot polling (~15ms per request, thread released immediately) supports all cameras with a 1-worker gunicorn.

### 17.8 Evidence accumulator (window=10, min_ratio=0.30 for phone)
**Why:** Single-frame detections cause alert flicker. But overhead custom model gives low confidence (20-38%) inconsistently. Solution: only need 3 out of 10 frames (30%) to confirm — catches real phone use while ignoring single-frame glitches.

### 17.9 `_force_red_box()` in saved photos
**Why:** The annotated frame from `annotate_cam()` uses evidence accumulator state which might be slightly ahead of the actual detection. Red box forces the violating person to be clearly marked even if the annotation timing doesn't match the detection timing perfectly.

### 17.10 Staggered camera startup (1.5s each)
**Why:** NVR rejects simultaneous connection requests. 4 cameras × 1.5s = 0-4.5s stagger ensures connections are sequential.

### 17.11 Phone validation: portrait + landscape (not portrait-only)
**Why:** v1 only accepted portrait orientation. Phones held horizontally at certain angles appear landscape. v2 accepts both with tightened bounds to exclude very-square objects.

### 17.12 Phone position zone: 5%-95% (not 20%-85%)
**Why:** v1 cut off phones held at ear level (top of person bbox). 5%-95% captures ear-level phone use.

---

## 18. Training Dataset — Current State

**Total images:** 85 (in `training_data/images/`)  
**Total label files:** 59 (some images may share labels or be negatives)  
**Cameras represented:** D1, D2, D6, D9, D19, D31, D44  
**Classes:**
```yaml
# training_data/dataset.yaml
names:
- phone_hand   # class 0
- phone_ear    # class 1
- phone_desk   # class 2
- sleeping     # class 3
- working      # class 4  (negative samples — empty label files)
nc: 5
path: /home/elsner/Documents/employees/training_data
train: images/train
val: images/val
```

**Verified folder:** `training_data/verified/` — copies of alert photos confirmed by OllamaVerifier as true positives

---

## 19. File Naming Conventions

| File pattern | Meaning |
|-------------|---------|
| `Cam{N}_{EVENT}_{YYYYMMDD_HHMMSS}.jpg` | Alert photo in logs/photos/ |
| `{cam}_{label}_{YYYYMMDD_HHMMSS_ffffff}.jpg` | Annotation image in training_data/images/ |
| `aug_{stem}_{op}.jpg` | Augmented image in training_data/augmented/images/ |
| `{cam_name}.jpg` | Shared frame in /tmp/monitor_frames/ |
| `{cam_name}.tmp.jpg` | Atomic temp file before rename |

---

## 20. How to Run From Scratch

### Prerequisites
```bash
# Python 3.8+ required
pip install ultralytics opencv-python python-dotenv numpy requests flask

# Optional (better accuracy):
pip install insightface onnxruntime-gpu   # GPU
pip install gunicorn                      # production web server
# Optional (AI descriptions):
ollama pull llava
```

### Model weights (auto-downloaded by ultralytics on first run, or manual):
- `yolov8s.pt` — https://github.com/ultralytics/assets/releases
- `yolov8n-pose.pt` — same source
- Place both in project root

### Configuration
```bash
cp cameras.json.example cameras.json
# Edit cameras.json with your camera IPs, credentials, channel numbers
# Edit .env with your thresholds and Ollama settings
```

### Start
```bash
chmod +x run.sh
./run.sh
# Opens: http://{local_ip}:5000
# Training UI: http://{local_ip}:5000/annotate
```

### Train custom model
1. Open `http://{ip}:5000/annotate`
2. Select camera, click Freeze when behavior visible
3. Draw box, select label, Save Annotation
4. Repeat 20+ times per class (targeting 60 per class)
5. Click 🚀 Train Model
6. Wait ~30min on GPU
7. Restart monitor.py (`./run.sh` again) to load new model

---

## 21. Shared State Boundaries — Thread Safety

| Shared object | Lock used | Writers | Readers |
|---------------|-----------|---------|---------|
| `cam.state.frame` | `cam.state._l` | `camera_reader` | `BatchDetectWorker`, `BatchPoseWorker`, `MotionWorker`, `FaceWorker`, `BehaviorEngine` |
| `cam.state.persons/phones/alerts` | `cam.state._l` | `BatchDetectWorker`, `BehaviorEngine` | `BatchPoseWorker`, `MotionWorker`, `FaceWorker`, `BehaviorEngine`, `annotate_cam()` |
| `cam.tracks` dict | `cam.tracks_lock` | `BatchDetectWorker`, `BatchPoseWorker`, `MotionWorker`, `FaceWorker`, `BehaviorEngine` | all of the above |
| `_detect_model`, `_custom_model`, `_pose_model` | `_infer_lock` | loaded once in main | `BatchDetectWorker`, `BatchPoseWorker` |
| `/tmp/monitor_frames/*.jpg` | atomic file rename | `camera_reader` (tmp→final) | `web_ui.CameraStream._poll` |
| `logs/logs.txt` | Python file `open("a")` | `rules.fire_alert()` | `web_ui /api/logs`, `OllamaVerifier` |
| `logs/photos/*.jpg` | none (write-once) | `rules.fire_alert()` | `web_ui /api/alerts`, `OllamaVerifier` |

---

## 22. Known Issues and Design Notes

1. **Custom model bboxes are person-sized:** The annotations drawn during training annotate around the entire person (not just the phone). This means `_valid_phone()` would reject them (wrong aspect ratio, wrong size). Solution: `skip_valid=True` for custom model detections.

2. **Overhead camera produces low-confidence phone detections (20-38%):** This is expected — the top-down angle is unusual for COCO-trained phone detection. The evidence accumulator with `min_ratio=0.30` (only 3/10 frames needed) compensates for this.

3. **InsightFace optional:** If not installed, `FaceWorker` silently disables itself. Sleep detection still works via `pose_sleeping` signal alone (requires `MOTION_STILL_SECS` to confirm without face signal).

4. **Ollama verifier removes false positives retroactively:** If Ollama says NO, photo is deleted and log entry removed. This improves dataset quality over time but requires `ollama pull llava` to be running.

5. **HEADLESS=true is the default:** Setting `HEADLESS=false` in `.env` enables a local `cv2.imshow()` display window (for developer machines with display). Not suitable for servers.

6. **Port 5000 is hardcoded** in web_ui.py and run.sh. Change both if needed.

7. **NVR connection limit:** `subtype=1` (sub-stream) allows more concurrent connections than `subtype=0` (main stream). The active cameras.json uses `subtype=0` — switch to `subtype=1` if NVR rejects connections.
