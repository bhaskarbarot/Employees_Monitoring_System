#!/usr/bin/env python3
"""
Employee Monitor v2 — Multi-Camera, GPU-Efficient
  • Single yolov8s instance shared across all cameras (load once)
  • Person + phone + chair detected in one inference pass
  • BatchDetectWorker + BatchPoseWorker — one GPU call per batch of cameras
  • fp16 half-precision + imgsz=416 — ~3x faster per inference
  • Simple IoU tracker per camera — no ByteTrack multi-camera conflicts
  • Phone validation fixed: portrait + landscape, ear zone 5-95%
  • HeadPoseWorker removed (redundant with PoseWorker yaw)
  • cameras.json for multi-camera; falls back to .env for single camera
"""

import os, time, signal, threading, json, urllib.parse, math
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2, torch, numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# ── Detection config ──────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
HALF        = DEVICE == "cuda"
CONF_PERSON = 0.50
CONF_PHONE  = 0.35          # balanced: custom model true positives are 35%+, false are 10-25%
CONF_CHAIR  = 0.40
CONF_KP     = 0.50
DISPLAY_W   = 1280
DISPLAY_H   = 720
PERSON, PHONE, CHAIR = 0, 67, 56

SLEEP_THRESHOLD      = int(os.getenv("SLEEP_THRESHOLD_SEC",       "120"))
# ── Session-based phone timing ─────────────────────────────────────────────
# Photo saved when session ENDS (not on detection start)
PHONE_SESSION_GRACE  = float(os.getenv("PHONE_SESSION_GRACE_SEC",  "6"))   # seconds without detection = session ended
PHONE_SESSION_MIN    = float(os.getenv("PHONE_SESSION_MIN_SEC",    "20"))   # ignore sessions shorter than this
PHONE_SESSION_MAX    = int(os.getenv("PHONE_SESSION_MAX_SEC",      "600"))  # periodic save every N sec during long sessions
MOTION_STILL_SECS    = 60
MOTION_THRESH        = 0.012
MIN_TRACK_AGE        = 30

# ── Camera-angle tuning (set in .env per camera) ──────────────────────────────
# CAMERA_ANGLE:
#   "side"     → camera mounted to the side of the person (profile view)
#                Normal = 1 ear visible.  BOTH ears visible = turned toward camera = wasting
#   "front"    → camera faces the person directly
#                Normal = 2 ears visible.  Only 1 ear visible = turned away = wasting
#   "overhead" → camera above, similar to front
CAMERA_ANGLE       = os.getenv("CAMERA_ANGLE",       "side").lower()
EAR_CONF_VISIBLE   = float(os.getenv("EAR_CONF_VISIBLE",   "0.50"))  # min conf to count ear as visible
EAR_CONF_HIDDEN    = float(os.getenv("EAR_CONF_HIDDEN",    "0.25"))  # max conf to count ear as hidden
STANDING_RATIO     = float(os.getenv("STANDING_RATIO",     "1.6"))   # bbox height/width > this = standing
LEANING_BACK_RATIO = float(os.getenv("LEANING_BACK_RATIO", "1.5"))   # nose above shoulders by N×shoulder_span

_running = True
def _stop(s=None, _=None): global _running; _running = False
signal.signal(signal.SIGINT,  _stop)
signal.signal(signal.SIGTERM, _stop)

# ── Global GPU models (loaded once, shared across all cameras) ────────────────
_detect_model  = None   # yolov8s.pt  — COCO person(0) + phone(67) detection
_custom_model  = None   # custom model — phone_hand/ear/desk/sleeping classifier
_pose_model    = None   # yolov8n-pose.pt
_infer_lock    = threading.Lock()

# Custom model class IDs (different from COCO)
CUSTOM_PHONE_HAND = 0
CUSTOM_PHONE_EAR  = 1
CUSTOM_PHONE_DESK = 2
CUSTOM_SLEEPING   = 3

def _load_models():
    global _detect_model, _custom_model, _pose_model
    dummy = np.zeros((416, 416, 3), dtype="uint8")

    # yolov8s.pt: ALWAYS used for person(COCO:0) + phone(COCO:67) detection
    # Custom model cannot replace this — it uses different class IDs
    print(f"  [Models] yolov8s.pt → {DEVICE}  fp16={HALF}  (COCO person+phone)")
    _detect_model = YOLO("yolov8s.pt")
    _detect_model([dummy], verbose=False, device=DEVICE, imgsz=416, half=HALF)

    # Custom model (if trained): runs on person CROPS for better classification
    custom = Path("custom_model/weights/best.pt")
    if custom.exists():
        print(f"  [Models] {custom} → {DEVICE}  (custom phone/sleep classifier)")
        _custom_model = YOLO(str(custom))
        _custom_model([dummy], verbose=False, device=DEVICE, imgsz=320, half=HALF)
        print(f"  [Models] *** Custom model active — better phone accuracy ***")
    else:
        print(f"  [Models] No custom model found — using COCO detection only")

    print(f"  [Models] yolov8n-pose.pt → {DEVICE}")
    _pose_model = YOLO("yolov8n-pose.pt")
    _pose_model([dummy], verbose=False, device=DEVICE, imgsz=416, half=HALF)
    print("  [Models] ready.\n")


# ══════════════════════════════════════════════════════════════════════════════
#  EVIDENCE ACCUMULATOR
# ══════════════════════════════════════════════════════════════════════════════

class EvidenceAcc:
    def __init__(self, window=8, min_ratio=0.70, min_frames=None):
        self._h = deque(maxlen=window); self._w = window; self._r = min_ratio
        # min_frames: how many frames must be seen before confirming
        # default = max(3, window//2) for slow behaviours
        # set low (e.g. 2) for fast/immediate detection like phone
        self._min = min_frames if min_frames is not None else max(3, window // 2)

    def add(self, v): self._h.append(1 if v else 0)
    def reset(self): self._h.clear()

    @property
    def confirmed(self):
        if len(self._h) < self._min: return False
        return (sum(self._h) / len(self._h)) >= self._r

    @property
    def ratio(self): return (sum(self._h) / len(self._h)) if self._h else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  PER-TRACK STATE
# ══════════════════════════════════════════════════════════════════════════════

class TrackState:
    def __init__(self, tid):
        self.tid = tid; self.bbox = None; self.centroid = (0, 0)
        self.last_seen = time.time(); self.first_seen = time.time()
        self.has_phone = False; self.phone_on_ear = False
        self.phone_type = None    # 'hand' | 'ear' | 'desk' | None
        self.phone_bbox = None    # full-frame bbox of detected phone
        self.pose_sleeping = False; self.face_visible = True
        self.motion_score = 1.0; self.is_still = False; self.still_since = None
        self.head_yaw = 0.0
        # wrist keypoints — used by phone classifier to confirm hand is on phone
        self.left_wrist  = None   # (x, y, conf)
        self.right_wrist = None   # (x, y, conf)
        # phone: 3 consecutive frames needed (~1.5s) — reduces flicker false positives
        # Custom model from overhead gives inconsistent detections (20-38% conf)
        # Window=10 frames (5s), need 3 positives = 30% ratio to confirm
        self.ev_phone = EvidenceAcc(window=10, min_ratio=0.30, min_frames=3)
        self.ev_sleep = EvidenceAcc(window=12, min_ratio=0.75)
        self.sleep_start       = None  # when sleeping session started
        self.sleep_last_active = None  # last time sleep_ok was True
        self.sleep_session_ann = None  # annotated frame at sleep start

        # ── Session-based timing (photo saved when session ENDS) ──────────
        self.phone_session_start  = None   # monotonic time when session started
        self.phone_session_ptype  = None   # 'hand' or 'ear' for this session
        self.phone_last_active    = None   # last monotonic time phone_ok was True
        self.phone_session_ann    = None   # annotated frame from peak of session
        self.phone_session_saved  = 0.0   # last periodic save time (for long sessions)
        self.phone_total_sec      = 0.0   # cumulative seconds across all sessions

    @property
    def track_age(self): return time.time() - self.first_seen

    @property
    def sleep_raw(self):
        # pose_sleeping is REQUIRED — no keypoints = no sleep alert
        # Kills false positives: helmets, bags, empty chairs, desks
        if not self.pose_sleeping:
            return False
        # Also need at least 1 more signal: face hidden OR body still for 60s
        supporting = [
            not self.face_visible,
            self.is_still and self.still_since is not None
            and (time.time() - self.still_since) > MOTION_STILL_SECS,
        ]
        return sum(supporting) >= 1

    @property
    def phone_raw(self):
        # phone_type requires YOLO to have actually detected a phone object
        # phone_on_ear alone (wrist near ear, no phone detected) = ignored
        # — stops "scratching head / adjusting glasses" false positives
        yolo_detected = self.phone_type in ('hand', 'ear')
        pose_confirmed = self.phone_on_ear and self.has_phone
        return yolo_detected or pose_confirmed


# ══════════════════════════════════════════════════════════════════════════════
#  PER-CAMERA STATE
# ══════════════════════════════════════════════════════════════════════════════

class CameraState:
    def __init__(self):
        self._l = threading.Lock()
        self.frame = None; self.persons = []; self.phones = []; self.alerts = []

    def set_frame(self, f):
        with self._l: self.frame = f

    def get_frame(self):
        with self._l: return self.frame

    def update(self, **kw):
        with self._l:
            for k, v in kw.items(): setattr(self, k, v)

    def snapshot(self):
        with self._l:
            return dict(frame=self.frame, persons=list(self.persons),
                        phones=list(self.phones), alerts=list(self.alerts))


# ══════════════════════════════════════════════════════════════════════════════
#  SIMPLE IoU TRACKER  (per-camera, no shared state — safe for multi-camera)
# ══════════════════════════════════════════════════════════════════════════════

class SimpleTracker:
    def __init__(self): self._next = 1; self._active = {}

    @staticmethod
    def _iou(a, b):
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        iw = max(0, ix2-ix1); ih = max(0, iy2-iy1); inter = iw * ih
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / max(ua, 1)

    def update(self, dets):
        """dets: [(bbox, conf)].  Returns [(bbox, conf, track_id)]."""
        for t in self._active.values(): t['age'] += 1
        results = []; matched = set()
        for bbox, conf in dets:
            best_iou = 0.30; best = None
            for tid, t in self._active.items():
                if tid in matched: continue
                iou = self._iou(bbox, t['bbox'])
                if iou > best_iou: best_iou = iou; best = tid
            if best:
                self._active[best].update(bbox=bbox, age=0)
                matched.add(best)
                results.append((bbox, conf, best))
            else:
                tid = self._next; self._next += 1
                self._active[tid] = {'bbox': bbox, 'age': 0}
                results.append((bbox, conf, tid))
        self._active = {t: v for t, v in self._active.items() if v['age'] < 45}
        return results


# ══════════════════════════════════════════════════════════════════════════════
#  PHONE VALIDATION  (fixed vs v1: landscape + ear zone included)
# ══════════════════════════════════════════════════════════════════════════════

def _classify_phone(phone_bbox, person_bbox, phone_on_ear_signal=False,
                    left_wrist=None, right_wrist=None):
    """
    Classify phone position.  Returns: 'ear' | 'hand' | 'desk'

    Logic:
      1. PoseWorker ear signal → 'ear'
      2. Phone in top 28% of person bbox → 'ear'
      3. Wrist keypoint is INSIDE or touching the phone bbox → 'hand'
      4. No wrist near phone → 'desk'  (phone lying on desk, not held)
    """
    if phone_on_ear_signal:
        return 'ear'

    px1, py1, px2, py2 = phone_bbox
    phone_cx = (px1 + px2) / 2
    phone_cy = (py1 + py2) / 2

    ex1, ey1, ex2, ey2 = person_bbox
    p_h = max(ey2 - ey1, 1)
    rel_y = (phone_cy - ey1) / p_h

    # Top 28% = head/ear zone
    if rel_y < 0.28:
        return 'ear'

    # Wrist proximity check.
    # 30px margin — balanced for overhead camera keypoint imprecision (~10-20px).
    # Previous 60% was too loose (desk phone triggered when typing nearby).
    # 12px was too tight (real phone holding missed from overhead angle).
    MARGIN = 30   # pixels — wrist must be within ~1 phone-width of phone

    for wrist in (left_wrist, right_wrist):
        if wrist is None: continue
        wx, wy, wc = wrist
        if wc < 0.30: continue
        if (px1 - MARGIN <= wx <= px2 + MARGIN and
                py1 - MARGIN <= wy <= py2 + MARGIN):
            return 'hand'

    # No wrist near phone → lying on desk, person not holding it
    return 'desk'


def _valid_phone(phone_bbox, person_bbox, frame_h, frame_w):
    px1, py1, px2, py2 = phone_bbox
    ph, pw = py2 - py1, px2 - px1
    if pw <= 0 or ph <= 0: return False
    aspect = ph / pw

    # Portrait (tall) OR landscape (wide) — v1 rejected landscape phones
    portrait  = 1.3 <= aspect <= 4.5   # tightened: exclude very square objects
    landscape = 0.22 <= aspect <= 0.77  # tightened: exclude very square objects
    if not (portrait or landscape): return False

    area = ph * pw; f_area = frame_h * frame_w
    if area < f_area * 0.0003: return False   # too small (mouse/pen)
    if area > f_area * 0.04:   return False   # too large (monitor/whiteboard)

    # Phone center must be in person's body zone 5%-95%
    # — v1 used 20%-85%, which cut off phones held at ear level
    ex1, ey1, ex2, ey2 = person_bbox; p_h = max(ey2 - ey1, 1)
    phone_cy = (py1 + py2) / 2
    if not (ey1 + p_h * 0.05 <= phone_cy <= ey1 + p_h * 0.95): return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA SESSION  (config + state + tracker per camera)
# ══════════════════════════════════════════════════════════════════════════════

class CameraSession:
    def __init__(self, cfg):
        self.cam_id  = cfg['id']
        self.name    = cfg.get('name', f"Cam-{cfg['id']}")
        self.cfg     = cfg
        self.state   = CameraState()
        self.tracks  = {}
        self.tracks_lock = threading.Lock()
        self.tracker = SimpleTracker()

    def rtsp_url(self):
        c = self.cfg
        pwd  = urllib.parse.quote(str(c.get('pass', '')), safe='')
        base = f"rtsp://{c.get('user','admin')}:{pwd}@{c['ip']}:{c.get('port',554)}"
        if c.get('rtsp_path'): return base + c['rtsp_path']
        ch      = c.get('channel', 1)
        subtype = c.get('subtype', 1)   # 0=main stream, 1=sub-stream (default)
        # subtype=1 allows more concurrent NVR connections than subtype=0
        if c.get('type', 'dahua').lower() == 'dahua':
            return base + f"/cam/realmonitor?channel={ch}&subtype={subtype}"
        return base + f"/Streaming/Channels/{ch}0{subtype+1}"


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH DETECT WORKER  — two-pass: persons on full frames, phones on crops
#
#  Key accuracy improvement:
#  Pass 1  full frame → imgsz=416  → finds persons (fast, batch)
#  Pass 2  person crops → imgsz=320  → finds phones (zoomed in, 4× more pixels)
# ══════════════════════════════════════════════════════════════════════════════

class BatchDetectWorker(threading.Thread):
    def __init__(self, cameras):
        super().__init__(daemon=True, name="BatchDetect")
        self.cameras = cameras

    def run(self):
        while _running:
            t0 = time.time()
            pairs = [(c, c.state.get_frame()) for c in self.cameras]
            pairs = [(c, f) for c, f in pairs if f is not None]
            if pairs:
                cams   = [c for c, f in pairs]
                frames = [f for c, f in pairs]
                try:
                    self._detect_batch(cams, frames)
                except Exception as e:
                    print(f"[BatchDetect] {e}")
            time.sleep(max(0, 0.5 - (time.time() - t0)))

    def _detect_batch(self, cams, frames):
        now = time.time()

        # ── Pass 1: persons on full frames (batch) ────────────────────────
        with _infer_lock:
            r1 = _detect_model(frames, verbose=False, conf=CONF_PERSON,
                               imgsz=416, device=DEVICE, half=HALF,
                               classes=[PERSON])

        persons_by_idx = []
        crop_meta = []   # (cam_idx, pe, crop, off_x, off_y, fh, fw)

        for i, (cam, r, frame) in enumerate(zip(cams, r1, frames)):
            fh, fw = frame.shape[:2]
            persons_raw = []
            if r.boxes:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf >= CONF_PERSON:
                        persons_raw.append((tuple(map(int, box.xyxy[0])), conf))

            tracked = cam.tracker.update(persons_raw)
            persons = []
            for bbox, conf, tid in tracked:
                x1, y1, x2, y2 = bbox; bw, bh = x2-x1, y2-y1
                standing = bw > 0 and bh / bw > STANDING_RATIO
                persons.append({'track_id': tid, 'bbox': bbox, 'conf': conf,
                                'centroid': ((x1+x2)//2, (y1+y2)//2)})
                with cam.tracks_lock:
                    if tid not in cam.tracks: cam.tracks[tid] = TrackState(tid)
                    ts = cam.tracks[tid]
                    ts.bbox = bbox; ts.centroid = ((x1+x2)//2, (y1+y2)//2)
                    ts.last_seen = now; ts.standing = standing

            persons_by_idx.append(persons)
            cam.state.update(persons=persons, phones=[])

            # Collect person crops — slight padding so hand area is included
            pad = 25
            for pe in persons:
                x1, y1, x2, y2 = pe['bbox']
                cx1 = max(0, x1-pad); cy1 = max(0, y1-pad)
                cx2 = min(fw, x2+pad); cy2 = min(fh, y2+pad)
                if cx2-cx1 < 30 or cy2-cy1 < 30: continue
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    crop_meta.append((i, pe, crop, cx1, cy1, fh, fw))

        if not crop_meta:
            self._purge(cams, now); return

        # ── Pass 2: phone detection on batched person crops ───────────────
        # Run BOTH models and merge results:
        #   Custom model: trained on your cameras → better hand/ear/desk classification
        #   Base COCO model: class 67 (phone) → strong general phone detector
        # Combining both gives best recall — custom model catches overhead phones
        # even at low confidence (20-38%), COCO catches what custom misses.
        crops = [m[2] for m in crop_meta]

        # Custom model results
        r2_custom = None
        if _custom_model is not None:
            with _infer_lock:
                r2_custom = _custom_model(crops, verbose=False, conf=CONF_PHONE,
                                          imgsz=320, device=DEVICE, half=HALF,
                                          classes=[CUSTOM_PHONE_HAND, CUSTOM_PHONE_EAR,
                                                   CUSTOM_PHONE_DESK])

        # COCO base model results (always run as fallback)
        with _infer_lock:
            r2_coco = _detect_model(crops, verbose=False, conf=0.25,
                                    imgsz=320, device=DEVICE, half=HALF,
                                    classes=[PHONE])

        phones_by_idx     = [[] for _ in cams]
        phone_tids_by_idx = [set() for _ in cams]

        def _add_phone(cam_idx, pe, ox, oy, fh, fw, pb, ptype, skip_valid=False):
            """Add a validated phone detection to the results."""
            pconf = float(pb.conf[0])
            if pconf < CONF_PHONE: return
            px1, py1, px2, py2 = map(int, pb.xyxy[0])
            full_bbox = (ox+px1, oy+py1, ox+px2, oy+py2)
            # Custom model: skip _valid_phone — it already classifies type correctly
            # (annotations drawn around person, not phone → bbox fails size checks)
            if not skip_valid and not _valid_phone(full_bbox, pe['bbox'], fh, fw): return
            # Never downgrade a confirmed hand/ear detection to desk
            existing = next((p for p in phones_by_idx[cam_idx]
                             if p['track_id'] == pe['track_id']), None)
            if existing and existing['type'] in ('hand','ear') and ptype == 'desk':
                return
            phones_by_idx[cam_idx].append({
                'bbox': full_bbox, 'conf': pconf,
                'track_id': pe['track_id'], 'type': ptype
            })
            cam = cams[cam_idx]
            with cam.tracks_lock:
                if pe['track_id'] in cam.tracks:
                    ts = cam.tracks[pe['track_id']]
                    if ptype in ('hand','ear') or ts.phone_type is None:
                        ts.phone_bbox = full_bbox
                        ts.phone_type = ptype
                        ts.has_phone  = ptype in ('hand','ear')

        # ── Process custom model (overhead-trained, knows hand/ear/desk) ──
        # skip_valid=True because custom model bboxes are person-sized
        # (annotations were drawn around the person, not the phone)
        if r2_custom is not None:
            for (cam_idx, pe, crop, ox, oy, fh, fw), pr in zip(crop_meta, r2_custom):
                if pr.boxes is None: continue
                for pb in pr.boxes:
                    cls_id = int(pb.cls[0])
                    if   cls_id == CUSTOM_PHONE_HAND: _add_phone(cam_idx, pe, ox, oy, fh, fw, pb, 'hand', skip_valid=True)
                    elif cls_id == CUSTOM_PHONE_EAR:  _add_phone(cam_idx, pe, ox, oy, fh, fw, pb, 'ear',  skip_valid=True)
                    elif cls_id == CUSTOM_PHONE_DESK: _add_phone(cam_idx, pe, ox, oy, fh, fw, pb, 'desk', skip_valid=True)

        # ── Process COCO base model (fills gaps custom model missed) ──────
        for (cam_idx, pe, crop, ox, oy, fh, fw), pr in zip(crop_meta, r2_coco):
            if pr.boxes is None: continue
            cam = cams[cam_idx]
            for pb in pr.boxes:
                # Get wrist signals for classification
                ear_sig = False; lw = None; rw = None
                with cam.tracks_lock:
                    if pe['track_id'] in cam.tracks:
                        ts0 = cam.tracks[pe['track_id']]
                        ear_sig = ts0.phone_on_ear
                        lw = ts0.left_wrist
                        rw = ts0.right_wrist
                # Compute full-frame phone bbox for classification
                px1, py1, px2, py2 = map(int, pb.xyxy[0])
                full_bbox = (ox+px1, oy+py1, ox+px2, oy+py2)
                ptype = _classify_phone(full_bbox, pe['bbox'], ear_sig, lw, rw)
                _add_phone(cam_idx, pe, ox, oy, fh, fw, pb, ptype)

                if ptype in ('hand', 'ear'):
                    phone_tids_by_idx[cam_idx].add(pe['track_id'])

        for cam, persons, phones in zip(cams, persons_by_idx, phones_by_idx):
            cam.state.update(phones=phones)
            # Clear phone state for persons where no phone was found this frame
            detected_tids = {p['track_id'] for p in phones}
            with cam.tracks_lock:
                for pe in persons:
                    if pe['track_id'] not in detected_tids and pe['track_id'] in cam.tracks:
                        ts = cam.tracks[pe['track_id']]
                        ts.has_phone = False; ts.phone_type = None; ts.phone_bbox = None

        self._purge(cams, now)

    def _purge(self, cams, now):
        for cam in cams:
            with cam.tracks_lock:
                stale = [t for t, ts in cam.tracks.items() if now - ts.last_seen > 900]
                for t in stale: del cam.tracks[t]


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH POSE WORKER  (keypoints for sleeping/posture, all cameras one call)
# ══════════════════════════════════════════════════════════════════════════════

class BatchPoseWorker(threading.Thread):
    def __init__(self, cameras):
        super().__init__(daemon=True, name="BatchPose")
        self.cameras = cameras

    def run(self):
        while _running:
            t0 = time.time()
            pairs = [(c, c.state.get_frame()) for c in self.cameras]
            pairs = [(c, f) for c, f in pairs if f is not None]
            if pairs:
                cams, frames = zip(*pairs)
                try:
                    with _infer_lock:
                        results = _pose_model(
                            list(frames), verbose=False, conf=CONF_PERSON,
                            imgsz=416, device=DEVICE, half=HALF
                        )
                    for cam, r, frame in zip(cams, results, frames):
                        persons = cam.state.snapshot()['persons']
                        if persons:
                            self._update(cam, r, persons, frame.shape[:2])
                except Exception as e:
                    print(f"[BatchPose] {e}")
            time.sleep(max(0, 1.5 - (time.time() - t0)))

    def _update(self, cam, r, persons, hw):
        frame_h, frame_w = hw
        for kp in (r.keypoints or []):
            if kp is None: continue
            xy    = kp.xy[0].cpu().numpy()
            confs = kp.conf[0].cpu().numpy()
            if xy[0][0] == 0 and xy[0][1] == 0: continue
            tid = self._match(xy[0][0], xy[0][1], persons)
            if tid is None: continue
            res = self._checks(xy, confs, frame_h, frame_w)
            with cam.tracks_lock:
                if tid in cam.tracks:
                    ts = cam.tracks[tid]
                    ts.pose_sleeping = res['sleeping']
                    ts.phone_on_ear  = res['phone_on_ear']
                    ts.head_yaw      = res['yaw_deg']
                    ts.left_wrist    = res['left_wrist']
                    ts.right_wrist   = res['right_wrist']

    def _match(self, nx, ny, persons):
        for pe in persons:
            x1, y1, x2, y2 = pe['bbox']
            if x1 <= nx <= x2 and y1 <= ny <= y2: return pe['track_id']
        return None

    def _checks(self, xy, confs, h, w):
        C = CONF_KP
        nose_x, nose_y, nc   = xy[0][0],  xy[0][1],  confs[0]
        le_x,   le_y,   lec  = xy[3][0],  xy[3][1],  confs[3]
        re_x,   re_y,   rec  = xy[4][0],  xy[4][1],  confs[4]
        ls_x,   ls_y,   lsc  = xy[5][0],  xy[5][1],  confs[5]
        rs_x,   rs_y,   rsc  = xy[6][0],  xy[6][1],  confs[6]
        lw_x,   lw_y,   lwc  = xy[9][0],  xy[9][1],  confs[9]
        rw_x,   rw_y,   rwc  = xy[10][0], xy[10][1], confs[10]

        valid_sh = [y for y, c in [(ls_y, lsc), (rs_y, rsc)] if c >= C]
        avg_sh   = sum(valid_sh) / len(valid_sh) if valid_sh else None

        # Use shoulder span as body-size reference for proportional thresholds
        if lsc >= C and rsc >= C:
            sh_span = max(abs(rs_x - ls_x), 40)
        elif lsc >= C or rsc >= C:
            sh_span = 60
        else:
            sh_span = 80

        sleeping = (nc >= C and avg_sh is not None
                    and nose_y > avg_sh + sh_span * 0.4)

        # Phone on ear: wrist within proportional distance of ear
        ear_phone = False
        thr_y = sh_span * 0.55; thr_x = sh_span * 0.65
        if lwc >= C and lec >= C:
            if abs(lw_y - le_y) < thr_y and abs(lw_x - le_x) < thr_x:
                ear_phone = True
        if rwc >= C and rec >= C:
            if abs(rw_y - re_y) < thr_y and abs(rw_x - re_x) < thr_x:
                ear_phone = True

        # Head yaw for display
        left_visible  = lec >= EAR_CONF_VISIBLE
        right_visible = rec >= EAR_CONF_VISIBLE
        if left_visible and right_visible:
            ear_mid  = (le_x + re_x) / 2
            ear_span = max(abs(re_x - le_x), 1)
            yaw_deg  = ((nose_x - ear_mid) / ear_span) * 90
        elif left_visible:  yaw_deg =  55.0
        elif right_visible: yaw_deg = -55.0
        else:               yaw_deg =   0.0

        # Wrist positions passed back so phone classifier can confirm hand is on phone
        lw_out = (lw_x, lw_y, lwc) if lwc >= 0.20 else None
        rw_out = (rw_x, rw_y, rwc) if rwc >= 0.20 else None

        return dict(sleeping=sleeping, phone_on_ear=ear_phone,
                    yaw_deg=yaw_deg,
                    left_wrist=lw_out, right_wrist=rw_out)


# ══════════════════════════════════════════════════════════════════════════════
#  MOTION WORKER  (per-camera, CPU-based)
# ══════════════════════════════════════════════════════════════════════════════

class MotionWorker(threading.Thread):
    def __init__(self, cam):
        super().__init__(daemon=True, name=f"Motion-{cam.cam_id}")
        self.cam = cam; self._prev = {}

    def run(self):
        while _running:
            t0 = time.time()
            frame = self.cam.state.get_frame()
            if frame is not None:
                try: self._process(frame.copy())
                except Exception as e: print(f"[{self.name}] {e}")
            time.sleep(max(0, 0.5 - (time.time() - t0)))

    def _process(self, frame):
        h, w = frame.shape[:2]; now = time.time()
        for pe in self.cam.state.snapshot()['persons']:
            tid = pe['track_id']; x1, y1, x2, y2 = pe['bbox']
            crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
            if crop.size == 0: continue
            gray = cv2.cvtColor(cv2.resize(crop, (64,64)), cv2.COLOR_BGR2GRAY)
            score = 1.0
            if tid in self._prev:
                score = cv2.absdiff(gray, self._prev[tid]).mean() / 255.0
            self._prev[tid] = gray
            with self.cam.tracks_lock:
                if tid in self.cam.tracks:
                    ts = self.cam.tracks[tid]; ts.motion_score = score
                    ts.is_still = (score < MOTION_THRESH)
                    if ts.is_still:
                        if ts.still_since is None: ts.still_since = now
                    else:
                        ts.still_since = None


# ══════════════════════════════════════════════════════════════════════════════
#  FACE WORKER  (optional InsightFace — face-visible signal for sleep)
# ══════════════════════════════════════════════════════════════════════════════

class FaceWorker(threading.Thread):
    def __init__(self, cam):
        super().__init__(daemon=True, name=f"Face-{cam.cam_id}")
        self.cam = cam; self._fa = None
        try:
            import insightface
            self._fa = insightface.app.FaceAnalysis(
                name="buffalo_sc",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                allowed_modules=["detection"])
            self._fa.prepare(ctx_id=0, det_size=(160, 160))
            print(f"  [Face-{cam.cam_id}] InsightFace ready")
        except Exception as e:
            print(f"  [Face-{cam.cam_id}] InsightFace unavailable ({e}), skipping")

    def run(self):
        while _running:
            t0 = time.time()
            frame = self.cam.state.get_frame()
            if frame is not None and self._fa:
                try: self._process(frame.copy())
                except Exception as e: print(f"[{self.name}] {e}")
            time.sleep(max(0, 2.0 - (time.time() - t0)))

    def _process(self, frame):
        h, w = frame.shape[:2]
        for pe in self.cam.state.snapshot()['persons']:
            tid = pe['track_id']; x1, y1, x2, y2 = pe['bbox']
            crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
            if crop.size < 100: continue
            try:
                faces   = self._fa.get(cv2.resize(crop, (160,160)))
                visible = len(faces) > 0
            except: visible = True
            with self.cam.tracks_lock:
                if tid in self.cam.tracks:
                    self.cam.tracks[tid].face_visible = visible


# ══════════════════════════════════════════════════════════════════════════════
#  OLLAMA VERIFIER  — removes false positive alert photos using llava vision
#  Enabled when USE_OLLAMA=true in .env
# ══════════════════════════════════════════════════════════════════════════════

class OllamaVerifier(threading.Thread):
    """
    Watches logs/photos/ for new alert images.
    Sends each to Ollama llava with a yes/no question.
    Deletes false positives and removes them from the log.
    Verified true positives are copied to training_data/ for future retraining.
    """

    QUESTIONS = {
        "PHONE_HAND": (
            "This is a CEILING/OVERHEAD security camera looking DOWN at an office. "
            "Look at the person inside the RED or ORANGE bounding box. "
            "Is that specific person CLEARLY holding a rectangular mobile phone "
            "in their hand? Look for a small rectangular device in their hand. "
            "Ignore people in GREEN boxes. "
            "Answer YES only if you can clearly see a phone in the highlighted person's hand. "
            "Answer NO if it is a mouse, keyboard, pen, or if no phone is visible. "
            "Reply with only YES or NO."
        ),
        "PHONE_EAR": (
            "This is a CEILING/OVERHEAD security camera looking DOWN at an office. "
            "Look at the person inside the RED or ORANGE bounding box. "
            "Is that specific person CLEARLY holding a mobile phone to their ear "
            "while talking? From overhead, this looks like a hand raised to the side "
            "of the head with a small rectangle visible. "
            "Do NOT answer YES if the person is just touching their face, "
            "scratching their head, or adjusting glasses. "
            "Answer YES only if a phone is clearly visible near their ear. "
            "Reply with only YES or NO."
        ),
        "SLEEPING": (
            "This is a CEILING/OVERHEAD security camera looking DOWN at an office. "
            "Look at the person inside the RED bounding box. "
            "Is that person's head resting DOWN on the desk or their arms, "
            "appearing to be asleep? From overhead, a sleeping person's head "
            "will be very close to the desk surface. "
            "Reply with only YES or NO."
        ),
    }

    def __init__(self):
        super().__init__(daemon=True, name="OllamaVerifier")
        self._seen     = set()
        self._photos   = Path("logs/photos")
        self._log_file = Path("logs/logs.txt")
        self._train_dir = Path("training_data")

    def run(self):
        from dotenv import load_dotenv; load_dotenv()
        if os.getenv("USE_OLLAMA", "false").lower() != "true":
            print("  [OllamaVerifier] disabled (USE_OLLAMA=false in .env)")
            return
        model = os.getenv("OLLAMA_MODEL", "llava:latest")
        print(f"  [OllamaVerifier] started — using {model}")

        while _running:
            try: self._scan(model)
            except Exception as e: print(f"  [OllamaVerifier] {e}")
            time.sleep(8)

    def _scan(self, model):
        if not self._photos.exists(): return
        photos = sorted(self._photos.glob("Cam*.jpg"),
                        key=lambda x: x.stat().st_mtime, reverse=True)[:15]
        for p in photos:
            if p.name in self._seen: continue
            self._seen.add(p.name)

            # Determine question based on event in filename
            q = None
            for event_key, question in self.QUESTIONS.items():
                if event_key in p.name:
                    q = question; break
            if q is None: continue

            answer = self._ask(str(p), q, model)

            if answer == "NO":
                print(f"  [OllamaVerifier] ✗ FALSE POSITIVE → removing {p.name}")
                try:
                    p.unlink()
                    self._remove_log_entry(p.name)
                except Exception as e:
                    print(f"  [OllamaVerifier] delete error: {e}")
            else:
                print(f"  [OllamaVerifier] ✓ confirmed {p.name}")
                self._save_to_training(p)

    def _ask(self, image_path, question, model):
        try:
            import requests, base64
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": question,
                      "images": [b64], "stream": False,
                      "options": {"temperature": 0.05, "num_predict": 6}},
                timeout=45
            )
            if r.ok:
                resp = r.json().get("response", "").strip().upper()
                # Strict: only YES if response starts with YES or is clearly YES
                # Anything uncertain → keep the photo (don't delete)
                first_word = resp.split()[0] if resp.split() else ""
                if first_word == "NO": return "NO"
                if first_word == "YES": return "YES"
        except Exception as e:
            print(f"  [OllamaVerifier] Ollama error: {e}")
        return "YES"   # default: keep on error / uncertain

    def _remove_log_entry(self, filename):
        if not self._log_file.exists(): return
        try:
            lines = self._log_file.read_text(errors="ignore").splitlines()
            kept  = [l for l in lines if filename not in l]
            self._log_file.write_text("\n".join(kept) + "\n")
        except Exception as e:
            print(f"  [OllamaVerifier] log update error: {e}")

    def _save_to_training(self, photo_path):
        """Copy confirmed alert to training_data/verify/ for future training."""
        try:
            out = self._train_dir / "verified"
            out.mkdir(parents=True, exist_ok=True)
            dest = out / photo_path.name
            if not dest.exists():
                import shutil; shutil.copy2(photo_path, dest)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  BEHAVIOR ENGINE  (per-camera state machine)
# ══════════════════════════════════════════════════════════════════════════════

def _force_red_box(ann_frame, ts, ptype, orig_frame):
    """
    Draw a guaranteed red box on the violator in the saved photo.
    Uses live_bbox (current detection position) — ts.bbox can be stale
    when tracker re-assigns IDs between frames.
    """
    # live_bbox is passed as the 5th argument from BehaviorEngine
    bbox = getattr(ts, '_live_bbox', None) or ts.bbox
    if bbox is None or ann_frame is None:
        return ann_frame
    try:
        fh, fw = ann_frame.shape[:2]
        if orig_frame is not None:
            oh, ow = orig_frame.shape[:2]
            sx, sy = fw / max(ow, 1), fh / max(oh, 1)
        else:
            sx, sy = 1.0, 1.0
        x1 = int(bbox[0] * sx); y1 = int(bbox[1] * sy)
        x2 = int(bbox[2] * sx); y2 = int(bbox[3] * sy)
        RED = (0, 0, 220)
        cv2.rectangle(ann_frame, (x1, y1), (x2, y2), RED, 3)
        if ptype == 'ear':
            lbl = "PHONE ON EAR"
        elif ptype == 'sleep':
            lbl = "SLEEPING"
        else:
            lbl = "PHONE IN HAND"
        (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ly = max(y1 - 4, lh + 4)
        cv2.rectangle(ann_frame, (x1, ly - lh - 3), (x1 + lw + 6, ly + 2), RED, -1)
        cv2.putText(ann_frame, lbl, (x1 + 2, ly - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    except Exception:
        pass
    return ann_frame


class BehaviorEngine(threading.Thread):
    def __init__(self, cam):
        super().__init__(daemon=True, name=f"Behavior-{cam.cam_id}")
        self.cam = cam
        from rules import fire_alert
        self._fire = fire_alert

    def run(self):
        while _running:
            t0 = time.time()
            frame = self.cam.state.get_frame()
            if frame is not None:
                try:
                    overlays = self._tick(frame.copy())
                    self.cam.state.update(alerts=overlays)
                except Exception as e: print(f"[{self.name}] {e}")
            time.sleep(max(0, 0.5 - (time.time() - t0)))

    def _tick(self, frame):
        now = time.time(); overlays = []
        with self.cam.tracks_lock: snap = dict(self.cam.tracks)

        # Get current live person positions from detection
        live_snap = self.cam.state.snapshot()
        # Map: track_id → current bbox (only persons VISIBLE RIGHT NOW)
        live_bbox_by_tid = {pe['track_id']: pe['bbox'] for pe in live_snap['persons']}

        for tid, ts in snap.items():
            ts.ev_phone.add(ts.phone_raw)
            ts.ev_sleep.add(ts.sleep_raw)
            phone_ok = ts.ev_phone.confirmed
            sleep_ok = ts.ev_sleep.confirmed

            # ── CRITICAL: only alert if person is CURRENTLY VISIBLE ──────────
            # Prevents stale tracks (person left/re-tracked) from firing alerts
            # at wrong positions (empty seats, wrong person)
            live_bbox = live_bbox_by_tid.get(tid)
            if live_bbox is None:
                # Person not in current frame — skip alert, keep accumulating
                continue

            # Use live detection bbox for accurate red box placement
            ts._live_bbox = live_bbox

            # ── PHONE ────────────────────────────────────────────────────────
            # TWO photos per event:
            #   1. IMMEDIATE — first frame when detected (shows it's happening)
            #   2. SESSION END — when phone put down (shows exact duration)
            if phone_ok:
                ptype = ts.phone_type or ('ear' if ts.phone_on_ear else 'hand')
                event = "PHONE_EAR" if ptype == 'ear' else "PHONE_HAND"

                with self.cam.tracks_lock:
                    if tid in self.cam.tracks:
                        t = self.cam.tracks[tid]
                        t.phone_last_active = now

                        if t.phone_session_start is None:
                            # ── NEW SESSION: save IMMEDIATE photo ────────────
                            t.phone_session_start = now
                            t.phone_session_ptype = ptype
                            t.phone_session_saved = now
                            ann_first = annotate_cam(self.cam)
                            ann_first = _force_red_box(ann_first, t, ptype,
                                                        self.cam.state.get_frame())
                            t.phone_session_ann = ann_first
                            # Fire immediately — "0 sec" photo shows detection start
                            self._fire(event, ann_first, self._det(t, "phone"),
                                       0, self.cam.cam_id, self.cam.name)
                        else:
                            # ── ONGOING SESSION: update best frame every 8s ──
                            if now - t.phone_session_saved >= 8:
                                ann_cur = annotate_cam(self.cam)
                                ann_cur = _force_red_box(ann_cur, t, ptype,
                                                          self.cam.state.get_frame())
                                t.phone_session_ann = ann_cur
                                t.phone_session_saved = now

                # Live overlay — show duration
                session_dur = now - (ts.phone_session_start or now)
                total_use   = ts.phone_total_sec + session_dur
                m_u, s_u    = int(total_use//60), int(total_use%60)
                label = "PHONE ON EAR" if ptype=='ear' else "PHONE IN HAND"
                overlays.append(f"[{self.cam.name}#{tid}] {label}  {m_u}m{s_u:02d}s")

                # Periodic save every PHONE_SESSION_MAX seconds (for long calls)
                if (ts.phone_session_start and session_dur >= PHONE_SESSION_MAX and
                        now - ts.phone_session_saved >= PHONE_SESSION_MAX):
                    ann = ts.phone_session_ann or annotate_cam(self.cam)
                    self._fire(event, ann, self._det(ts, "phone"), session_dur,
                               self.cam.cam_id, self.cam.name)
                    with self.cam.tracks_lock:
                        if tid in self.cam.tracks:
                            self.cam.tracks[tid].phone_session_saved = now

            else:
                # Phone gone — check if session just ended
                with self.cam.tracks_lock:
                    if tid in self.cam.tracks:
                        t = self.cam.tracks[tid]
                        if (t.phone_session_start is not None and
                                t.phone_last_active is not None and
                                now - t.phone_last_active >= PHONE_SESSION_GRACE):
                            # ── SESSION ENDED: save final photo with duration ─
                            duration = t.phone_last_active - t.phone_session_start
                            t.phone_total_sec += duration
                            ptype = t.phone_session_ptype or 'hand'
                            event = "PHONE_EAR" if ptype == 'ear' else "PHONE_HAND"
                            if duration >= PHONE_SESSION_MIN:
                                ann = t.phone_session_ann or annotate_cam(self.cam)
                                self._fire(event, ann, self._det(t, "phone"),
                                           duration, self.cam.cam_id, self.cam.name)
                            t.phone_session_start = None
                            t.phone_last_active   = None
                            t.phone_session_ann   = None
                            t.phone_session_ptype = None

            # ═══════════════════════════════════════════════════════════════
            #  SLEEPING — two photos per session:
            #    Photo 1: IMMEDIATE when sleeping confirmed (after threshold)
            #    Photo 2: FINAL when they wake up with exact duration
            # ═══════════════════════════════════════════════════════════════
            if sleep_ok:
                with self.cam.tracks_lock:
                    if tid in self.cam.tracks:
                        t = self.cam.tracks[tid]
                        t.sleep_last_active = now
                        if t.sleep_start is None:
                            t.sleep_start = now

                        elapsed = now - t.sleep_start
                        # ── PHOTO 1: immediate after threshold confirmed ───────
                        if elapsed >= SLEEP_THRESHOLD and t.sleep_session_ann is None:
                            ann = annotate_cam(self.cam)
                            ann = _force_red_box(ann, t, 'sleep', self.cam.state.get_frame())
                            t.sleep_session_ann = ann
                            self._fire("SLEEPING", ann, self._det(t, "sleep"),
                                       elapsed, self.cam.cam_id, self.cam.name)

                elapsed = now - (ts.sleep_start or now)
                overlays.append(f"[{self.cam.name}#{tid}] SLEEPING "
                                 f"{int(elapsed//60)}m{int(elapsed%60):02d}s "
                                 f"({min(elapsed/SLEEP_THRESHOLD*100,100):.0f}%)")
            else:
                with self.cam.tracks_lock:
                    if tid in self.cam.tracks:
                        t = self.cam.tracks[tid]
                        if (t.sleep_start is not None and
                                t.sleep_last_active is not None and
                                t.sleep_session_ann is not None):
                            # ── PHOTO 2: final — they woke up ─────────────────
                            duration = t.sleep_last_active - t.sleep_start
                            if duration >= SLEEP_THRESHOLD:
                                self._fire("SLEEPING", t.sleep_session_ann,
                                           self._det(t, "sleep"),
                                           duration, self.cam.cam_id, self.cam.name)
                        t.sleep_start       = None
                        t.sleep_last_active = None
                        t.sleep_session_ann = None



        return overlays

    def _det(self, ts, vtype):
        snap = self.cam.state.snapshot()
        return {
            'persons':        [{'bbox': pe['bbox'], 'conf': pe['conf']} for pe in snap['persons']],
            'phones':         [{'bbox': ph['bbox'], 'conf': ph['conf']} for ph in snap['phones']],
            'phone_violator': ts.bbox if vtype == 'phone' else None,
            'sleep_violator': ts.bbox if vtype == 'sleep' else None,
            'person_present': len(snap['persons']) > 0,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA READER THREAD
# ══════════════════════════════════════════════════════════════════════════════

SHARED_FRAMES_DIR = Path("/tmp/monitor_frames")
SHARED_FRAMES_DIR.mkdir(exist_ok=True)

def camera_reader(cam, startup_delay=0):
    """
    startup_delay: stagger camera startups by N seconds so the NVR doesn't
    receive all 7 connection requests simultaneously (avoids timeout/rejection).
    Writes latest frame to /tmp/monitor_frames/{name}.jpg so web_ui
    can read it without opening its own RTSP connection.
    """
    if startup_delay > 0:
        time.sleep(startup_delay)

    url = cam.rtsp_url()
    print(f"  [Cam-{cam.cam_id}] {cam.name}  →  {url}")

    retry_delay = 3
    while _running:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print(f"  [Cam-{cam.cam_id}] Cannot open — retry in {retry_delay}s")
            cap.release()
            time.sleep(retry_delay)
            retry_delay = min(retry_delay + 2, 15)
            continue

        retry_delay = 3   # reset on success
        _frame_count = 0
        while _running:
            ret, frame = cap.read()
            if not ret:
                print(f"  [Cam-{cam.cam_id}] Stream lost — reconnecting in 3s...")
                break
            cam.state.set_frame(frame)
            # Share frame with web_ui every 3rd frame (~3fps write)
            # so web_ui never needs its own RTSP connection
            _frame_count += 1
            if _frame_count % 3 == 0:
                try:
                    shared = cv2.resize(frame, (640, 360))
                    tmp = SHARED_FRAMES_DIR / f"{cam.name}.tmp.jpg"
                    cv2.imwrite(str(tmp), shared, [cv2.IMWRITE_JPEG_QUALITY, 72])
                    tmp.rename(SHARED_FRAMES_DIR / f"{cam.name}.jpg")
                except Exception:
                    pass

        cap.release()
        if _running:
            time.sleep(3)


# ══════════════════════════════════════════════════════════════════════════════
#  ANNOTATION
# ══════════════════════════════════════════════════════════════════════════════

_G = (0,200,0); _R = (0,0,220); _Y = (0,200,220); _O = (0,140,255)

def annotate_cam(cam, target_w=None, target_h=None):
    frame = cam.state.get_frame()
    if frame is None:
        h = target_h or DISPLAY_H; w = target_w or DISPLAY_W
        blank = np.zeros((h, w, 3), dtype="uint8")
        cv2.putText(blank, f"{cam.name} — no signal", (20, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80,80,80), 2, cv2.LINE_AA)
        return blank
    frame = frame.copy()
    snap  = cam.state.snapshot()
    with cam.tracks_lock: track_snap = dict(cam.tracks)
    h, w  = frame.shape[:2]

    # Build set of track IDs that currently have an active phone detection
    active_phone_tids = {ph['track_id'] for ph in snap['phones']
                         if ph.get('type') in ('hand', 'ear')}

    for pe in snap['persons']:
        tid = pe['track_id']; ts = track_snap.get(tid)
        x1,y1,x2,y2 = pe['bbox']
        if ts is None:
            color, label = _G, "OK"
        elif ts.ev_phone.confirmed or tid in active_phone_tids:
            # Use live phone detection OR confirmed evidence — whichever is current
            ptype = ts.phone_type or ('ear' if ts.phone_on_ear else 'hand')
            if ptype == 'ear': color, label = (0,60,255), "PHONE ON EAR"
            else:              color, label = _R,          "PHONE IN HAND"
        elif ts.ev_sleep.confirmed:
            color, label = _R, "SLEEPING"
        else:
            color, label = _G, "OK"
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        (lw2,lh2),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        ly = max(y1-4, lh2+4)
        cv2.rectangle(frame, (x1, ly-lh2-3), (x1+lw2+4, ly+2), color, -1)
        cv2.putText(frame, label, (x1+2, ly-1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 2, cv2.LINE_AA)
        if ts:
            # Show track ID + cumulative phone usage time per person
            total_phone = ts.phone_total_sec
            if ts.phone_session_start is not None:
                total_phone += time.time() - ts.phone_session_start
            if total_phone > 5:
                m_p, s_p = int(total_phone//60), int(total_phone%60)
                info_txt = f"#{tid}  Phone: {m_p}m{s_p:02d}s"
            else:
                info_txt = f"#{tid}"
            cv2.putText(frame, info_txt, (x1+2, y2-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1, cv2.LINE_AA)

    for ph in snap['phones']:
        x1,y1,x2,y2 = ph['bbox']
        ptype = ph.get('type', 'hand')
        if   ptype == 'ear':  col, lbl = (0, 60, 255), "PHONE ON EAR"
        elif ptype == 'desk': col, lbl = (100,100,100), "PHONE ON DESK"
        else:                 col, lbl = _O,            "PHONE IN HAND"
        cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
        cv2.putText(frame, lbl, (x1+2, max(y1-4, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 2, cv2.LINE_AA)

    for i, a in enumerate(snap['alerts']):
        cv2.putText(frame, a, (10, 28+i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,0,230), 2, cv2.LINE_AA)

    n_p  = len(snap['persons']); n_ph = len(snap['phones'])
    n_s  = sum(1 for ts in track_snap.values() if ts.ev_sleep.confirmed)
    bar  = (f"{cam.name} | {datetime.now():%H:%M:%S} | "
            f"Persons:{n_p}  Phone:{n_ph}  Sleeping:{n_s}")
    cv2.rectangle(frame, (0, h-26), (w, h), (30,30,30), -1)
    cv2.putText(frame, bar, (8, h-7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1, cv2.LINE_AA)

    if target_w and target_h:
        frame = cv2.resize(frame, (target_w, target_h))
    return frame


def make_grid(cameras):
    """Arrange all camera feeds in a grid at DISPLAY_W×DISPLAY_H."""
    n    = len(cameras)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    cw   = DISPLAY_W // cols
    ch   = DISPLAY_H // rows
    grid = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype="uint8")
    for i, cam in enumerate(cameras):
        r, c = divmod(i, cols)
        y1, y2 = r*ch, (r+1)*ch; x1, x2 = c*cw, (c+1)*cw
        grid[y1:y2, x1:x2] = annotate_cam(cam, cw, ch)
    return grid


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _load_camera_configs():
    if os.path.exists("cameras.json"):
        with open("cameras.json") as f:
            cfgs = json.load(f)
        print(f"[INFO] Loaded {len(cfgs)} cameras from cameras.json")
        return cfgs
    print("[INFO] cameras.json not found — using single camera from .env")
    return [{
        "id":       1,
        "name":     "Cam-1",
        "ip":       os.getenv("CAMERA_IP",        "192.168.30.5"),
        "user":     os.getenv("CAMERA_USER",       "admin"),
        "pass":     os.getenv("CAMERA_PASS",       ""),
        "port":     int(os.getenv("CAMERA_PORT",   "554")),
        "channel":  int(os.getenv("CAMERA_CHANNEL","2")),
        "type":     os.getenv("CAMERA_TYPE",       "dahua"),
        "rtsp_path": os.getenv("CAMERA_RTSP_PATH", "").strip(),
    }]


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    gpu  = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
    vram = (torch.cuda.get_device_properties(0).total_memory // 1024 // 1024
            if DEVICE == "cuda" else 0)

    print(f"\n{'═'*65}")
    print(f"  GPU          : {gpu}  ({vram} MB)")
    print(f"  Mode         : fp16={HALF}  imgsz=416  batch-inference")
    print(f"  Detect       : yolov8s  (person + phone in one pass)")
    print(f"  Pose         : yolov8n-pose")
    print(f"  Phone zones  : portrait 1.2-5.0 + landscape 0.2-0.83, zone 5-95%")
    print(f"  Camera angle : {CAMERA_ANGLE.upper()}  "
          f"({'both ears = looking away' if CAMERA_ANGLE=='side' else 'one ear = looking away'})")
    print(f"  Standing     : bbox h/w > {STANDING_RATIO}")
    print(f"  Leaning back : nose > {LEANING_BACK_RATIO}x shoulder-span above shoulders")
    print(f"{'═'*65}\n")

    print("[INFO] Loading shared models...")
    _load_models()

    cfgs    = _load_camera_configs()
    cameras = [CameraSession(c) for c in cfgs]
    n       = len(cameras)
    print(f"[INFO] Initializing {n} camera(s)...\n")

    # Start RTSP reader threads — staggered by 1.5s each
    # Prevents the NVR from receiving all 7 connection requests simultaneously
    for i, cam in enumerate(cameras):
        delay = i * 1.5   # 0s, 1.5s, 3s, 4.5s, 6s, 7.5s, 9s
        t = threading.Thread(target=camera_reader, args=(cam, delay), daemon=True)
        t.start()

    # Wait for at least one live frame
    print("[INFO] Waiting for cameras...", end="", flush=True)
    for _ in range(150):
        if any(c.state.get_frame() is not None for c in cameras): break
        time.sleep(0.1)
    live = sum(1 for c in cameras if c.state.get_frame() is not None)
    print(f" {live}/{n} live.")

    # Shared batch workers
    w_detect = BatchDetectWorker(cameras)
    w_pose   = BatchPoseWorker(cameras)
    w_detect.start(); w_pose.start()
    print(f"  ✓ BatchDetectWorker  (yolov8s, all {n} cams)")
    print(f"  ✓ BatchPoseWorker    (yolov8n-pose, all {n} cams)")

    # Per-camera workers
    for cam in cameras:
        MotionWorker(cam).start()
        FaceWorker(cam).start()
        BehaviorEngine(cam).start()
        print(f"  ✓ Workers for {cam.name}")

    # Ollama false-positive verifier (runs if USE_OLLAMA=true)
    OllamaVerifier().start()
    print(f"  ✓ OllamaVerifier")

    # HEADLESS mode — no local display window needed.
    # The web dashboard at :5000 is the UI. cv2.imshow caused
    # "force quit" dialogs on Wayland/headless servers.
    HEADLESS = os.getenv("HEADLESS", "true").lower() != "false"
    print(f"\n[INFO] Monitoring {n} camera(s). Headless={HEADLESS}. Ctrl+C to stop.\n")

    if HEADLESS:
        # Production mode: no display window, just keep running
        while _running:
            time.sleep(1)
    else:
        # Developer mode: show local window (set HEADLESS=false in .env)
        use_grid = n > 1
        while _running:
            if use_grid:
                display = make_grid(cameras)
                cv2.imshow(f"Employee Monitor — {n} cameras  [Q=quit]", display)
            else:
                display = annotate_cam(cameras[0])
                display = cv2.resize(display, (DISPLAY_W, DISPLAY_H))
                cv2.imshow("Employee Monitor  [Q=quit]", display)
            if cv2.waitKey(33) & 0xFF == ord('q'): _stop()
        cv2.destroyAllWindows()

    time.sleep(0.3)
    print("[INFO] Stopped.")


if __name__ == "__main__":
    main()
