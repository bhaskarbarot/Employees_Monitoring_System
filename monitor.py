#!/usr/bin/env python3
"""
Employee Monitor v3 — Multi-Camera, GPU-Efficient, Production-Grade
  • Single yolov8s instance shared across all cameras (load once)
  • Person + phone + chair detected in one inference pass
  • BatchDetectWorker + BatchPoseWorker — one GPU call per batch of cameras
  • fp16 half-precision + imgsz=416 — ~3x faster per inference
  • KalmanTracker per camera — stable IDs, Hungarian assignment, no ID flicker
  • Phone validation: portrait + landscape, ear zone 5-95%
  • HeadPoseWorker removed (redundant with PoseWorker yaw)
  • cameras.json for multi-camera; falls back to .env for single camera
  • CLAHE+EMA motion worker — robust to monitor flicker and lighting changes
  • Structured Python logging — filterable, timestamped, level-aware
"""

import logging
import os, time, signal, threading, json, urllib.parse, math
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2, torch, numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

# ── Structured logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-14s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
_log = logging.getLogger("monitor")

# ── v4 component imports (lazy — failures degrade gracefully) ────────────────
try:
    from tracker import KalmanTracker as _KalmanTrackerV4, DeskZoneReID
    _TRACKER_V4 = True
    _log.info("[v4] tracker.py loaded — ByteTrack-hybrid + DeskZoneReID active")
except ImportError:
    _TRACKER_V4 = False
    _log.info("[v4] tracker.py not found — using built-in KalmanTracker")

try:
    from fusion import PhoneFusion, SleepFusion
    _FUSION_V4 = True
    _log.info("[v4] fusion.py loaded — probabilistic FusionEngine active")
except ImportError:
    _FUSION_V4 = False
    _log.info("[v4] fusion.py not found — using EvidenceAcc")

try:
    from auto_label import AutoLabelService
    _AUTOLABEL_V4 = True
except ImportError:
    _AUTOLABEL_V4 = False

try:
    from mining import HardNegativeMiner
    _MINING_V4 = True
except ImportError:
    _MINING_V4 = False

# Global hard-negative miner (shared across cameras)
_hard_neg_miner: Optional["HardNegativeMiner"] = None

# Global desk-zone re-ID registry (shared across cameras)
_desk_reid: Optional["DeskZoneReID"] = None

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
PHONE_SESSION_MIN    = float(os.getenv("PHONE_SESSION_MIN_SEC",    "5"))    # save photos for sessions >= 5s (was 20s)
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

    def _load_one(pt_path: str, engine_path: str, imgsz: int, label: str):
        """Load TensorRT engine if available, else fall back to .pt model."""
        eng = Path(engine_path)
        pt  = Path(pt_path)
        if eng.exists() and DEVICE == "cuda":
            try:
                _log.info("[Models] %s → TensorRT engine (2-4× faster)", label)
                m = YOLO(str(eng))
                m([dummy], verbose=False, device=DEVICE, imgsz=imgsz)
                return m
            except Exception as exc:
                _log.warning("[Models] TRT load failed (%s), using .pt: %s", label, exc)
        _log.info("[Models] %s → PyTorch .pt  fp16=%s", label, HALF)
        m = YOLO(str(pt))
        m([dummy], verbose=False, device=DEVICE, imgsz=imgsz, half=HALF)
        return m

    _detect_model = _load_one("yolov8s.pt", "yolov8s.engine", 416, "yolov8s (person+phone)")

    custom_pt  = Path("custom_model/weights/best.pt")
    custom_eng = Path("custom_model/weights/best.engine")
    if custom_pt.exists():
        _custom_model = _load_one(str(custom_pt), str(custom_eng), 320, "custom classifier")
        _log.info("[Models] *** Custom model active — better phone accuracy ***")
    else:
        _log.info("[Models] No custom model found — using COCO detection only")

    _pose_model = _load_one("yolov8n-pose.pt", "yolov8n-pose.engine", 416, "yolov8n-pose")
    _log.info("[Models] ready.")


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
        # Legacy frame-ratio accumulators (v3 — kept for backward compat)
        self.ev_phone = EvidenceAcc(window=10, min_ratio=0.20, min_frames=3)
        self.ev_sleep = EvidenceAcc(window=12, min_ratio=0.75)

        # v4 probabilistic fusion engines (active if fusion.py imported)
        if _FUSION_V4:
            self.phone_fusion = PhoneFusion()
            self.sleep_fusion  = SleepFusion()
        else:
            self.phone_fusion = None
            self.sleep_fusion  = None

        # Alert cooldown tracking (v4 deduplication)
        self.last_alert_end_time  : float = 0.0   # monotonic time of last session end
        self.last_alert_event_type: Optional[str] = None

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
        # YOLO path: phone object detected in hand or at ear
        yolo_detected = self.phone_type in ('hand', 'ear')

        # Pose path: wrist-near-ear from keypoints, standalone signal.
        #
        # BUG FIX (v3.1): previously gated by has_phone (required YOLO to also
        # detect a phone object).  From an overhead camera during a phone call,
        # the phone is completely hidden between the hand and the head — YOLO
        # detects 0 phones at any confidence threshold.  The has_phone gate
        # was silently suppressing every overhead phone-call alert.
        #
        # False-positive risk of standalone pose signal:
        # Head-touching / hair adjustment is brief (1-3 frames, <1s).
        # EvidenceAcc requires min_frames=3 AND sustained ratio over 10 frames,
        # PLUS BehaviorEngine requires PHONE_SESSION_MIN=20s before saving a
        # photo — so brief gestures do NOT produce alerts.
        pose_ear_only = self.phone_on_ear
        return yolo_detected or pose_ear_only


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
#  TRACKER UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _iou_matrix(boxes_a: List[Tuple], boxes_b: List[Tuple]) -> np.ndarray:
    """Compute N×M IoU matrix between two lists of (x1,y1,x2,y2) boxes."""
    n, m = len(boxes_a), len(boxes_b)
    mat  = np.zeros((n, m), dtype=np.float32)
    for i, a in enumerate(boxes_a):
        ax1, ay1, ax2, ay2 = a
        a_area = max((ax2 - ax1) * (ay2 - ay1), 1e-6)
        for j, b in enumerate(boxes_b):
            ix1 = max(ax1, b[0]); iy1 = max(ay1, b[1])
            ix2 = min(ax2, b[2]); iy2 = min(ay2, b[3])
            iw  = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            union = a_area + (b[2]-b[0])*(b[3]-b[1]) - inter
            mat[i, j] = inter / max(union, 1e-6)
    return mat


# ══════════════════════════════════════════════════════════════════════════════
#  KALMAN BOX  —  single-object Kalman filter for bbox tracking
# ══════════════════════════════════════════════════════════════════════════════

class KalmanBox:
    """
    Kalman filter for a single bounding-box track.

    State  X = [cx, cy, w, h, vx, vy]ᵀ   (constant-velocity model, dt=1 frame)
    Measurement Z = [cx, cy, w, h]ᵀ

    Noise parameters are tuned for overhead office cameras (1280×720, ~25fps):
    people sit mostly stationary; YOLO bboxes have ±4-8px jitter.

    Benefits over raw IoU matching:
    • Predicts where a person WILL be before the next frame arrives,
      so IoU stays high even during brief movement → no ID switch.
    • Smoothed bbox reduces jitter in annotation and red-box placement.
    • Handles short occlusions (person briefly hidden) without ID reset.
    """

    _NOISE_POS  = 3.0    # position process noise  (px/frame)
    _NOISE_VEL  = 8.0    # velocity process noise  (px/frame²)
    _NOISE_MEAS = 6.0    # YOLO measurement noise  (px), covers ~±6px jitter

    def __init__(self, bbox: Tuple[int, int, int, int]) -> None:
        cx, cy, w, h = self._to_cxcywh(bbox)

        # State transition matrix  (cx += vx·dt,  cy += vy·dt,  dt=1)
        self.F = np.eye(6, dtype=np.float32)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0

        # Measurement matrix  (observe position; velocity is latent)
        self.H = np.zeros((4, 6), dtype=np.float32)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

        # Process noise Q
        p2 = self._NOISE_POS ** 2
        v2 = self._NOISE_VEL ** 2
        self.Q = np.diag([p2, p2, p2 * 0.5, p2 * 0.5, v2, v2]).astype(np.float32)

        # Measurement noise R  (w/h jitter is larger than cx/cy jitter)
        r2 = self._NOISE_MEAS ** 2
        self.R = np.diag([r2, r2, r2 * 3.0, r2 * 3.0]).astype(np.float32)

        # Initial state
        self.x = np.array([cx, cy, w, h, 0.0, 0.0], dtype=np.float32)

        # Initial covariance  (high uncertainty on velocity; moderate on position)
        self.P = np.diag([p2 * 4, p2 * 4, p2 * 8, p2 * 8,
                          v2 * 50, v2 * 50]).astype(np.float32)

        self.age              : int = 0   # total frames since creation
        self.time_since_update: int = 0   # frames since last measurement
        # Start at 1: this box was born from a real detection (counts as first hit)
        self.hit_streak       : int = 1   # consecutive matched frames

    # ── prediction step ───────────────────────────────────────────────────────

    def predict(self) -> Tuple[int, int, int, int]:
        """Advance one time-step.  Returns predicted (x1,y1,x2,y2)."""
        self.x[2] = max(self.x[2], 10.0)   # prevent degenerate bbox width
        self.x[3] = max(self.x[3], 10.0)   # prevent degenerate bbox height
        self.x    = self.F @ self.x
        self.P    = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0             # broke the streak
        self.time_since_update += 1
        return self._to_xyxy(self.x[:4])

    # ── correction step ───────────────────────────────────────────────────────

    def update(self, bbox: Tuple[int, int, int, int]) -> None:
        """Correct state with a new YOLO measurement."""
        cx, cy, w, h = self._to_cxcywh(bbox)
        z = np.array([cx, cy, w, h], dtype=np.float32)

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(6, dtype=np.float32) - K @ self.H) @ self.P

        self.time_since_update = 0
        self.hit_streak       += 1

    def get_state(self) -> Tuple[int, int, int, int]:
        """Return current smoothed bbox as (x1,y1,x2,y2)."""
        return self._to_xyxy(self.x[:4])

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_cxcywh(bbox: Tuple) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        return (x1+x2)/2.0, (y1+y2)/2.0, float(x2-x1), float(y2-y1)

    @staticmethod
    def _to_xyxy(s: np.ndarray) -> Tuple[int, int, int, int]:
        cx, cy, w, h = s
        return (int(cx - w/2), int(cy - h/2),
                int(cx + w/2), int(cy + h/2))


# ══════════════════════════════════════════════════════════════════════════════
#  KALMAN TRACKER  — production-grade multi-object tracker
#
#  Replaces SimpleTracker.  Identical interface:
#      tracker.update([(bbox, conf), ...]) → [(bbox, conf, track_id), ...]
#
#  Improvements over SimpleTracker:
#  • Kalman prediction prevents ID switching during brief movement / occlusion
#  • Hungarian algorithm (scipy) gives globally optimal assignment vs. greedy
#  • Track lifecycle: new → tentative → confirmed → lost → removed
#  • Smoothed bboxes reduce jitter in overlays, red-box placement, crop padding
#  • max_age=45 matches SimpleTracker's age < 45 removal policy exactly
# ══════════════════════════════════════════════════════════════════════════════

class KalmanTracker:
    """
    Production-grade multi-object Kalman tracker with Hungarian assignment.

    Parameters
    ----------
    max_age       : frames to keep an unmatched track before deletion (same as
                    SimpleTracker's age < 45 threshold → default 45)
    min_hits      : consecutive matched frames before a track is returned.
                    Set to 1 for backward compatibility — EvidenceAcc provides
                    the real confirmation window; single-frame FPs don't fire alerts.
    iou_threshold : minimum IoU to accept an assignment (lowered vs SimpleTracker's
                    0.30 because Kalman prediction keeps predicted boxes closer to
                    true position — 0.20 gives better recall during movement)
    """

    def __init__(self, max_age: int = 45, min_hits: int = 1,
                 iou_threshold: float = 0.20) -> None:
        self._next_id : int                    = 1
        self._boxes   : Dict[int, KalmanBox]  = {}
        self._confs   : Dict[int, float]      = {}
        self.max_age  = max_age
        self.min_hits = min_hits
        self.iou_thr  = iou_threshold

    def update(self, dets: List[Tuple]) -> List[Tuple]:
        """
        Parameters
        ----------
        dets : [(bbox, conf), ...]   bbox = (x1, y1, x2, y2)

        Returns
        -------
        [(bbox, conf, track_id), ...]  — only active confirmed tracks
        Smoothed bbox from Kalman filter, not raw YOLO bbox.
        """
        # ── Step 1: predict all active tracks ────────────────────────────────
        tids        : List[int]              = list(self._boxes.keys())
        predictions : List[Tuple[int, ...]] = [self._boxes[t].predict() for t in tids]

        # ── Step 2: associate predictions ↔ detections ────────────────────────
        matched, unmatched_dets = self._associate(predictions, tids, dets)

        # ── Step 3: update matched tracks ────────────────────────────────────
        for tid, det_idx in matched:
            self._boxes[tid].update(dets[det_idx][0])
            self._confs[tid] = dets[det_idx][1]

        # ── Step 4: spawn new tracks for unmatched detections ─────────────────
        for det_idx in unmatched_dets:
            bbox, conf = dets[det_idx]
            new_tid = self._next_id; self._next_id += 1
            self._boxes[new_tid] = KalmanBox(bbox)
            self._confs[new_tid] = conf

        # ── Step 5: remove stale tracks ───────────────────────────────────────
        stale = [t for t, kb in self._boxes.items()
                 if kb.time_since_update > self.max_age]
        for t in stale:
            del self._boxes[t]
            self._confs.pop(t, None)

        # ── Step 6: return confirmed, active tracks ───────────────────────────
        results: List[Tuple] = []
        for tid, kb in self._boxes.items():
            if kb.time_since_update == 0 and kb.hit_streak >= self.min_hits:
                results.append((kb.get_state(), self._confs[tid], tid))
        return results

    # ── internal ──────────────────────────────────────────────────────────────

    def _associate(
        self,
        predictions : List[Tuple],
        tids        : List[int],
        dets        : List[Tuple],
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        Hungarian assignment gated by IoU threshold.

        Returns
        -------
        matched        : [(track_id, det_index), ...]
        unmatched_dets : [det_index, ...]
        """
        if not predictions or not dets:
            return [], list(range(len(dets)))

        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            # scipy unavailable — fall back to greedy (same as SimpleTracker)
            return self._greedy_associate(predictions, tids, dets)

        det_bboxes = [d[0] for d in dets]
        iou_mat    = _iou_matrix(predictions, det_bboxes)
        cost_mat   = 1.0 - iou_mat

        row_ind, col_ind = linear_sum_assignment(cost_mat)

        matched      : List[Tuple[int, int]] = []
        matched_dets : set                   = set()
        for r, c in zip(row_ind, col_ind):
            if iou_mat[r, c] >= self.iou_thr:
                matched.append((tids[r], c))
                matched_dets.add(c)

        unmatched_dets = [i for i in range(len(dets)) if i not in matched_dets]
        return matched, unmatched_dets

    def _greedy_associate(self, predictions, tids, dets):
        """Fallback greedy match used when scipy is unavailable."""
        det_bboxes  = [d[0] for d in dets]
        iou_mat     = _iou_matrix(predictions, det_bboxes)
        matched      : List[Tuple[int, int]] = []
        matched_dets : set                   = set()
        matched_trks : set                   = set()
        for r in range(len(predictions)):
            best_c, best_iou = -1, self.iou_thr
            for c in range(len(det_bboxes)):
                if c in matched_dets: continue
                if iou_mat[r, c] > best_iou:
                    best_iou = iou_mat[r, c]; best_c = c
            if best_c >= 0:
                matched.append((tids[r], best_c))
                matched_dets.add(best_c); matched_trks.add(r)
        unmatched_dets = [i for i in range(len(dets)) if i not in matched_dets]
        return matched, unmatched_dets


# ══════════════════════════════════════════════════════════════════════════════
#  SIMPLE IoU TRACKER  (kept for reference — replaced by KalmanTracker above)
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

    # Accept any phone-like aspect ratio: landscape, square, or portrait.
    # Previous tightened ranges (0.22-0.77 OR 1.3-4.5) had a gap 0.77-1.3 that
    # rejected real phone detections when COCO drew a square bbox (aspect≈1.0).
    # Only reject extreme slivers (<0.15) or extremely wide panels (>5.0).
    if not (0.15 <= aspect <= 5.0): return False

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
        self.tracks  : Dict[int, "TrackState"] = {}
        self.tracks_lock = threading.RLock()  # RLock: annotate_cam re-acquires this inside BehaviorEngine's lock hold

        # ── Per-camera calibration profile (Phase 5) ─────────────────────────
        # Profile keys override global .env defaults for this specific camera.
        # If 'profile' key absent from cameras.json → all globals used (backward compat).
        prof = cfg.get('profile', {})
        self.prof_motion_thresh    = float(prof.get('motion_thresh',    MOTION_THRESH))
        self.prof_ear_thr_y_mult   = float(prof.get('ear_thr_y_mult',   0.06))
        self.prof_ear_thr_x_mult   = float(prof.get('ear_thr_x_mult',   0.08))
        self.prof_sleep_sh_mult    = float(prof.get('sleep_shoulder_mult', 0.4))
        self.prof_expected_persons = int(prof.get('expected_persons',   -1))   # -1 = unknown

        # ── Tracker (Phase 3.1 — v4 ByteTrack-hybrid, v3 KalmanTracker fallback) ──
        if _TRACKER_V4:
            self.tracker = _KalmanTrackerV4(max_age=45, min_hits=1, iou_threshold=0.20)
        else:
            self.tracker = KalmanTracker(max_age=45, min_hits=1, iou_threshold=0.20)

        # ── Metrics for health endpoint (Phase 6) ─────────────────────────────
        self.detect_latency_ms : float = 0.0   # last batch detect latency
        self.fps               : float = 0.0   # approximate streaming fps
        self._fps_ts           : float = time.time()
        self._fps_count        : int   = 0

    def record_frame(self) -> None:
        """Update FPS counter on each new frame from camera_reader."""
        self._fps_count += 1
        elapsed = time.time() - self._fps_ts
        if elapsed >= 5.0:
            self.fps       = self._fps_count / elapsed
            self._fps_count = 0
            self._fps_ts   = time.time()

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
    # Phase 4.2 — motion-gated inference
    # Cameras where ALL tracked persons have motion_score below this threshold
    # for N consecutive cycles are skipped to save GPU time.
    _MOTION_GATE_THRESH = 0.008   # below = completely still
    _MOTION_GATE_CYCLES = 1       # skip after this many consecutive still cycles
    _FORCE_EVERY        = 3       # force full inference every N cycles regardless

    def __init__(self, cameras):
        super().__init__(daemon=True, name="BatchDetect")
        self.cameras = cameras
        # Phase 4.2: per-camera still-cycle counter and forced-cycle counter
        self._still_cycles : Dict[int, int] = {}   # cam_id → consecutive still cycles
        self._force_counter: int            = 0

    def run(self):
        while _running:
            t0 = time.time()
            pairs = [(c, c.state.get_frame()) for c in self.cameras]
            pairs = [(c, f) for c, f in pairs if f is not None]
            if pairs:
                # ── Phase 4.2: motion-gated camera selection ──────────────────
                self._force_counter = (self._force_counter + 1) % self._FORCE_EVERY
                force_all = (self._force_counter == 0)

                active_pairs = []
                for c, f in pairs:
                    if force_all:
                        active_pairs.append((c, f))
                        self._still_cycles[c.cam_id] = 0
                        continue
                    # Check if ALL persons on this camera are completely still
                    with c.tracks_lock:
                        scores = [ts.motion_score for ts in c.tracks.values()]
                    all_still = scores and all(s < self._MOTION_GATE_THRESH for s in scores)
                    if all_still:
                        cnt = self._still_cycles.get(c.cam_id, 0) + 1
                        self._still_cycles[c.cam_id] = cnt
                        if cnt > self._MOTION_GATE_CYCLES:
                            # Skip this camera — persons haven't moved
                            continue
                    else:
                        self._still_cycles[c.cam_id] = 0
                    active_pairs.append((c, f))

                if active_pairs:
                    cams   = [c for c, f in active_pairs]
                    frames = [f for c, f in active_pairs]
                    t_infer = time.time()
                    try:
                        self._detect_batch(cams, frames)
                        latency = (time.time() - t_infer) * 1000
                        for cam in cams:
                            cam.detect_latency_ms = latency / len(cams)
                    except Exception as exc:
                        _log.error("[BatchDetect] %s", exc, exc_info=True)
            time.sleep(max(0, 0.5 - (time.time() - t0)))

    def _detect_batch(self, cams, frames):
        now = time.time()

        # ── Resolution normalisation (CRITICAL for high-res cameras) ─────────
        # At 2560×1440, person crops fed to the custom model (imgsz=320) are
        # 600×700+ pixels — phones vanish when letterboxed to 320. The model
        # was trained on crops from 640×360 frames where crops are ~160px wide.
        # Fix: downscale any frame wider than 1280 to 1280×720 before detection.
        # Person detection bbox coordinates are then in 1280×720 space; the
        # offsets passed to crop_meta are also in that space so phone bboxes
        # are correctly mapped back via the scale factor stored below.
        DET_MAX_W = 1280
        proc_frames = []
        frame_scales = []   # (sx, sy) — original / proc
        for frame in frames:
            oh, ow = frame.shape[:2]
            if ow > DET_MAX_W:
                sx = ow / DET_MAX_W; sy = oh / (DET_MAX_W * oh // ow)
                proc = cv2.resize(frame, (DET_MAX_W, DET_MAX_W * oh // ow))
            else:
                sx = sy = 1.0; proc = frame
            proc_frames.append(proc)
            frame_scales.append((sx, sy))

        # ── Pass 1: persons on full frames (batch) ────────────────────────
        with _infer_lock:
            r1 = _detect_model(proc_frames, verbose=False, conf=CONF_PERSON,
                               imgsz=416, device=DEVICE, half=HALF,
                               classes=[PERSON])

        persons_by_idx = []
        crop_meta = []   # (cam_idx, pe, crop, off_x, off_y, fh, fw)

        for i, (cam, r, frame) in enumerate(zip(cams, r1, proc_frames)):
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
                centroid = ((x1+x2)//2, (y1+y2)//2)
                persons.append({'track_id': tid, 'bbox': bbox, 'conf': conf,
                                'centroid': centroid})
                with cam.tracks_lock:
                    if tid not in cam.tracks:
                        # ── Phase 3.2: DeskZoneReID — restore session on return ──
                        ts_new = TrackState(tid)
                        if _desk_reid is not None:
                            old_tid = _desk_reid.query(centroid, tid, cam.cam_id)
                            if old_tid is not None and old_tid in cam.tracks:
                                old_ts = cam.tracks[old_tid]
                                ts_new.phone_total_sec     = old_ts.phone_total_sec
                                ts_new.ev_phone            = old_ts.ev_phone
                                ts_new.ev_sleep            = old_ts.ev_sleep
                                if _FUSION_V4 and old_ts.phone_fusion:
                                    ts_new.phone_fusion = old_ts.phone_fusion
                                    ts_new.sleep_fusion = old_ts.sleep_fusion
                                ts_new.last_alert_end_time   = old_ts.last_alert_end_time
                                ts_new.last_alert_event_type = old_ts.last_alert_event_type
                                del cam.tracks[old_tid]
                                _log.debug("[ReID] cam%d: merged track %d → %d",
                                           cam.cam_id, old_tid, tid)
                        cam.tracks[tid] = ts_new
                    ts = cam.tracks[tid]
                    ts.bbox = bbox; ts.centroid = centroid
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

        def _add_phone(cam_idx, pe, ox, oy, fh, fw, pb, ptype, skip_valid=False, min_conf=None):
            """Add a validated phone detection to the results."""
            pconf = float(pb.conf[0])
            threshold = min_conf if min_conf is not None else CONF_PHONE
            if pconf < threshold: return
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
                _add_phone(cam_idx, pe, ox, oy, fh, fw, pb, ptype, min_conf=0.25)

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
                    # Downscale high-res frames — same as BatchDetectWorker
                    proc = []
                    for f in frames:
                        oh, ow = f.shape[:2]
                        proc.append(cv2.resize(f, (1280, 1280*oh//ow)) if ow > 1280 else f)

                    with _infer_lock:
                        results = _pose_model(
                            proc, verbose=False, conf=CONF_PERSON,
                            imgsz=416, device=DEVICE, half=HALF
                        )
                    for cam, r, pframe in zip(cams, results, proc):
                        persons = cam.state.snapshot()['persons']
                        if persons:
                            self._update(cam, r, persons, pframe.shape[:2])
                except Exception as exc:
                    _log.error("[BatchPose] %s", exc, exc_info=True)
            time.sleep(max(0, 1.5 - (time.time() - t0)))

    def _update(self, cam, r, persons, hw):
        frame_h, frame_w = hw
        for kp in (r.keypoints or []):
            if kp is None: continue
            xy    = kp.xy[0].cpu().numpy()
            confs = kp.conf[0].cpu().numpy()
            if xy[0][0] == 0 and xy[0][1] == 0: continue
            tid = self._match(xy[0][0], xy[0][1], persons, frame_h, frame_w)
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

    def _match(self, nx, ny, persons, frame_h=720, frame_w=1280):
        """
        Match a pose keypoint set (identified by nose position) to a tracked person.

        Primary:  nose (nx, ny) must be inside the detection bbox.
        Fallback: nearest centroid within MAX_DIST pixels.

        BUG FIX (v3.2): At very high resolutions (2560×1440 tested), the
        YOLOv8-pose model's letterboxing and the YOLOv8s detection model's
        letterboxing differ slightly, causing the pose nose to land just outside
        the detection bbox (e.g., nose_y=182 vs bbox top=303 at 1440p).
        Without the fallback, _match() returns None → phone_on_ear is never
        set on any track → phone-call alerts never fire at that resolution.

        MAX_DIST scales with the smaller frame dimension (15%) so it stays
        proportional: 216px@2560×1440, 108px@1280×720, 54px@640×360.
        """
        # Primary: nose inside bbox
        for pe in persons:
            x1, y1, x2, y2 = pe['bbox']
            if x1 <= nx <= x2 and y1 <= ny <= y2:
                return pe['track_id']

        # Fallback: nearest centroid (handles letterbox offset between models)
        MAX_DIST = min(frame_h, frame_w) * 0.15   # ~15% of smaller dimension
        best_dist = MAX_DIST; best_tid = None
        for pe in persons:
            cx, cy = pe['centroid']
            dist = ((nx - cx) ** 2 + (ny - cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist; best_tid = pe['track_id']
        return best_tid

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
        #
        # BUG FIX (v3.1): from overhead cameras, shoulders are mostly hidden —
        # sh_span always clamps to 40px minimum → thr_y=22px, thr_x=26px.
        # These are too tight: a wrist 25px from ear (clearly a phone call) fails.
        #
        # Root cause: absolute 40px sh_span clamp was designed for side-view cameras.
        # From overhead the shoulder span in 2D is near-zero (collapsed depth axis).
        #
        # Fix: add frame-relative MINIMUMS so thresholds scale correctly at any
        # camera resolution (640×360, 1280×720, 1920×1080).
        #   thr_y_min = 6% of frame height  → 22px@360, 43px@720, 65px@1080
        #   thr_x_min = 8% of frame width   → 51px@640, 102px@1280, 154px@1920
        # Validated on live D6 1080p overhead footage:
        #   phone-caller wrist-ear dy≈57px, dx≈126px → both pass at 1080p
        #   non-caller   wrist-ear dy≈86px            → blocked by dy>thr_y
        ear_phone = False
        thr_y = max(sh_span * 0.55, h * 0.06)   # was sh_span*0.55 = 22px (too tight)
        thr_x = max(sh_span * 0.65, w * 0.08)   # was sh_span*0.65 = 26px (too tight)
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
    """
    Per-camera motion detection worker.

    Improvements over v2:
    • CLAHE normalization on the L channel (LAB space) before diff.
      This makes the diff invariant to monitor flicker, auto-exposure shifts,
      and fluorescent-light flicker — all major sources of false sleep detections.
    • Gaussian blur on the diff image suppresses JPEG block-noise and minor
      camera-shake artefacts that don't indicate real human motion.
    • EMA (exponential moving average) smoothing dampens single-frame spikes
      caused by compression or transient lighting events.
    • Active-set cleanup removes stale track IDs from internal state dicts,
      preventing unbounded memory growth when track IDs roll over.
    """

    # EMA alpha: 0 = perfectly smooth (slow), 1 = no smoothing (instant).
    # 0.35 means ~3 frames to respond to real motion; spike artefacts are
    # heavily damped.  Chosen so a sitting-still person is stable within ~1s.
    _EMA_ALPHA = 0.35

    def __init__(self, cam: "CameraSession") -> None:
        super().__init__(daemon=True, name=f"Motion-{cam.cam_id}")
        self.cam    = cam
        self._prev  : Dict[int, np.ndarray] = {}   # tid → normalised 64×64 L-channel
        self._scores: Dict[int, float]      = {}   # tid → EMA motion score
        # CLAHE normalises uneven lighting: clipLimit=2 avoids over-amplification
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    def run(self) -> None:
        while _running:
            t0 = time.time()
            frame = self.cam.state.get_frame()
            if frame is not None:
                try:
                    self._process(frame.copy())
                except Exception as exc:
                    _log.warning("[%s] %s", self.name, exc)
            time.sleep(max(0.0, 0.5 - (time.time() - t0)))

    def _process(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        now  = time.time()
        snap = self.cam.state.snapshot()

        active_tids: set = set()
        for pe in snap['persons']:
            tid = pe['track_id']
            active_tids.add(tid)
            x1, y1, x2, y2 = pe['bbox']
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size == 0:
                continue

            # ── CLAHE-normalised luminance crop (64×64) ───────────────────
            # Resize BEFORE CLAHE — cheaper and avoids tile-size edge effects
            small = cv2.resize(crop, (64, 64))
            lab   = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
            l_ch, _, _ = cv2.split(lab)
            norm  = self._clahe.apply(l_ch)   # lighting-normalised luminance

            # ── pixel diff with Gaussian denoising ───────────────────────
            raw_score = 1.0
            if tid in self._prev:
                diff = cv2.absdiff(norm, self._prev[tid])
                # 3×3 Gaussian removes JPEG-block and minor tremor noise
                diff = cv2.GaussianBlur(diff, (3, 3), 0)
                raw_score = float(diff.mean()) / 255.0
            self._prev[tid] = norm

            # ── EMA smoothing ─────────────────────────────────────────────
            prev_ema  = self._scores.get(tid, raw_score)
            ema_score = self._EMA_ALPHA * raw_score + (1.0 - self._EMA_ALPHA) * prev_ema
            self._scores[tid] = ema_score

            with self.cam.tracks_lock:
                if tid in self.cam.tracks:
                    ts = self.cam.tracks[tid]
                    ts.motion_score = ema_score
                    ts.is_still     = (ema_score < MOTION_THRESH)
                    if ts.is_still:
                        if ts.still_since is None:
                            ts.still_since = now
                    else:
                        ts.still_since = None

        # ── cleanup stale IDs to prevent memory leak ─────────────────────────
        stale = [k for k in list(self._prev.keys()) if k not in active_tids]
        for k in stale:
            self._prev.pop(k, None)
            self._scores.pop(k, None)


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
            "Overhead office security camera (ceiling view, looking down). "
            "There is a person with a RED or ORANGE bounding box drawn around them. "
            "Task: Is that specific person holding a mobile phone in their hand RIGHT NOW?\n"
            "- YES if you see a small rectangular device (phone/smartphone) in their hand\n"
            "- NO if their hand holds a mouse, pen, wallet, cup, or nothing visible\n"
            "- NO if you are unsure — only YES when clearly visible\n"
            "Ignore people in GREEN boxes. Reply with only YES or NO."
        ),
        "PHONE_EAR": (
            "Overhead office security camera (ceiling view, looking down). "
            "There is a person with a RED or ORANGE bounding box drawn around them. "
            "Task: Is that person holding a mobile phone to their ear/head making a phone call?\n"
            "- YES if their hand is raised to the side of their head with a phone\n"
            "- NO if they are just touching their face, scratching, or adjusting hair/glasses\n"
            "- NO if you cannot clearly see a phone near their head\n"
            "Ignore people in GREEN boxes. Reply with only YES or NO."
        ),
        "SLEEPING": (
            "Overhead office security camera (ceiling view, looking down). "
            "There is a person with a RED bounding box drawn around them. "
            "Task: Is that person's head resting DOWN on the desk surface, appearing to be asleep?\n"
            "- YES if their head is clearly down on the desk or arms (asleep/unconscious posture)\n"
            "- NO if they are just leaning forward, typing, reading, or looking down\n"
            "Reply with only YES or NO."
        ),
    }

    def __init__(self):
        super().__init__(daemon=True, name="OllamaVerifier")
        self._seen     = set()
        self._photos   = Path("logs/photos")
        self._log_file = Path("logs/logs.txt")
        self._train_dir = Path("training_data")

    # Vision-capable models in priority order (Qwen2.5-VL is best)
    _VISION_MODELS = [
        'qwen2.5vl', 'qwen2-vl', 'qwen2.5-vl',  # Qwen2.5-VL — best accuracy
        'llava',                                    # LLaVA — widely used
        'bakllava', 'llava-llama3', 'llava-phi3',  # LLaVA variants
        'moondream', 'moondream2',                  # Lightweight option
        'minicpm-v', 'cogvlm',                      # Other vision models
    ]

    def _find_vision_model(self) -> Optional[str]:
        """Find best available vision model from Ollama. Prefers Qwen2.5-VL."""
        import requests
        configured = os.getenv("OLLAMA_MODEL", "llava:latest")
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=5)
            if not r.ok:
                print("  [OllamaVerifier] Ollama not responding at :11434")
                return None
            available = [m['name'] for m in r.json().get('models', [])]
            # Try configured model first
            if any(configured.split(':')[0].lower() in a.lower() for a in available):
                return next(a for a in available
                            if configured.split(':')[0].lower() in a.lower())
            # Try vision models in priority order
            for vm in self._VISION_MODELS:
                match = next((a for a in available if vm in a.lower()), None)
                if match:
                    print(f"  [OllamaVerifier] auto-selected vision model: {match}")
                    return match
            print("  [OllamaVerifier] No vision model found in Ollama.")
            print("  [OllamaVerifier] Install one: ollama pull qwen2.5vl:7b")
            print("  [OllamaVerifier]         or: ollama pull llava")
            return None
        except Exception as e:
            print(f"  [OllamaVerifier] Cannot reach Ollama: {e}")
            return None

    def run(self):
        from dotenv import load_dotenv; load_dotenv()
        if os.getenv("USE_OLLAMA", "false").lower() == "false":
            print("  [OllamaVerifier] disabled (set USE_OLLAMA=true in .env to enable)")
            return
        model = self._find_vision_model()
        if not model:
            return
        print(f"  [OllamaVerifier] started — model={model} — scanning photos every 8s")

        while _running:
            try: self._scan(model)
            except Exception as e: print(f"  [OllamaVerifier] {e}")
            time.sleep(8)

    def _scan(self, model: str) -> None:
        if not self._photos.exists():
            return

        photos = sorted(self._photos.glob("Cam*.jpg"),
                        key=lambda x: x.stat().st_mtime, reverse=True)[:15]

        # ── Prune _seen: remove names of files that no longer exist on disk ──
        # Without this, _seen grows unboundedly: OllamaVerifier deletes the file
        # (answer == NO) but the name stays in _seen forever, so any file that is
        # later written with the same timestamp string is silently skipped.
        existing_names = {p.name for p in self._photos.glob("Cam*.jpg")}
        self._seen &= existing_names   # set intersection-update

        for p in photos:
            if p.name in self._seen:
                continue
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

        live_snap = self.cam.state.snapshot()
        live_bbox_by_tid = {pe['track_id']: pe['bbox'] for pe in live_snap['persons']}

        for tid, ts in snap.items():
            # ── Update legacy EvidenceAcc (always maintained for compatibility) ─
            ts.ev_phone.add(ts.phone_raw)
            ts.ev_sleep.add(ts.sleep_raw)

            # ── Update FusionEngine (v4 — if available) ───────────────────────
            if _FUSION_V4 and ts.phone_fusion is not None:
                # Feed YOLO detection confidence into phone fusion
                yolo_conf = 0.0
                if ts.phone_type in ('hand', 'ear') and ts.phone_bbox is not None:
                    # Use stored YOLO confidence from TrackState (best available)
                    yolo_conf = getattr(ts, '_last_phone_conf', 0.5)
                ts.phone_fusion.update_yolo(ts.phone_type, yolo_conf)
                ts.phone_fusion.update_pose(ts.phone_on_ear)
                ts.sleep_fusion.update(ts.pose_sleeping, not ts.face_visible, ts.is_still)

            # ── Determine confirmed state ──────────────────────────────────────
            # v4: FusionEngine takes priority when available and confirmed;
            # falls back to EvidenceAcc for backward compatibility.
            if _FUSION_V4 and ts.phone_fusion is not None and ts.phone_fusion.confirmed:
                phone_ok = True
                # Use fusion-determined type when it differs from YOLO
                if ts.phone_type is None:
                    ts.phone_type = ts.phone_fusion.dominant_type
            else:
                phone_ok = ts.ev_phone.confirmed

            if _FUSION_V4 and ts.sleep_fusion is not None and ts.sleep_fusion.confirmed:
                sleep_ok = True
            else:
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
                            # ── Phase 6: Alert deduplication ─────────────────
                            # If same event ended < 60s ago for this track,
                            # extend the old session rather than create a new alert
                            _DEDUP_COOLDOWN = 60.0
                            recent_same = (
                                t.last_alert_event_type == event and
                                now - t.last_alert_end_time < _DEDUP_COOLDOWN
                            )
                            if not recent_same:
                                # ── NEW SESSION: save IMMEDIATE photo ────────
                                # Set session_ptype BEFORE annotate_cam so the
                                # box label uses the correct type (EAR vs HAND).
                                t.phone_session_start = now
                                t.phone_session_ptype = ptype   # ← FIRST
                                t.phone_session_saved = now
                                ann_first = annotate_cam(self.cam)  # sees ptype above
                                ann_first = _force_red_box(ann_first, t, ptype,
                                                           self.cam.state.get_frame())
                                t.phone_session_ann = ann_first
                                self._fire(event, ann_first, self._det(t, "phone"),
                                           0, self.cam.cam_id, self.cam.name)
                            else:
                                # Resume: re-open session silently (no duplicate photo)
                                t.phone_session_start = t.last_alert_end_time
                                t.phone_session_ptype = ptype
                                t.phone_session_saved = now
                                t.phone_session_ann   = None
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
                            # Phase 6: record session end for deduplication
                            t.last_alert_end_time   = t.phone_last_active
                            t.last_alert_event_type = event
                            # Phase 3.2: register position for desk re-ID
                            if _desk_reid is not None and t.centroid:
                                _desk_reid.register_lost(tid, t.centroid, self.cam.cam_id)
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

def camera_reader(cam: "CameraSession", startup_delay: float = 0) -> None:
    """
    RTSP reader thread for one camera.

    Startup delay staggers connection attempts so the NVR never receives all
    cameras simultaneously (avoids timeout / connection-refused bursts).

    Frame sharing: every 3rd frame is written atomically to
    /tmp/monitor_frames/{name}.jpg so web_ui reads it without opening a
    second RTSP connection to the NVR.

    Reconnect policy (exponential backoff, v3):
    • First failure: retry after _RETRY_INIT seconds (2s)
    • Each subsequent failure: delay × _RETRY_MULT (×1.5) up to _RETRY_MAX (30s)
    • Successful open: resets delay to _RETRY_INIT
    • Consecutive read failures: require _FAIL_THRESHOLD bad reads before
      declaring stream lost — absorbs transient RTSP packet drops without
      unnecessary reconnects that spike NVR load.
    """
    _RETRY_INIT      = 2.0
    _RETRY_MULT      = 1.5
    _RETRY_MAX       = 30.0
    _FAIL_THRESHOLD  = 5        # consecutive bad reads before reconnect

    if startup_delay > 0:
        time.sleep(startup_delay)

    url = cam.rtsp_url()
    _log.info("[Cam-%d] %s  →  %s", cam.cam_id, cam.name, url)

    retry_delay      = _RETRY_INIT
    _frame_count     = 0

    while _running:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            _log.warning("[Cam-%d] Cannot open — retry in %.0fs", cam.cam_id, retry_delay)
            cap.release()
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * _RETRY_MULT, _RETRY_MAX)
            continue

        retry_delay       = _RETRY_INIT   # reset on successful open
        consecutive_fails = 0
        _log.info("[Cam-%d] %s — stream open", cam.cam_id, cam.name)

        while _running:
            ret, frame = cap.read()
            if not ret:
                consecutive_fails += 1
                if consecutive_fails >= _FAIL_THRESHOLD:
                    _log.warning("[Cam-%d] %d consecutive read failures — reconnecting",
                                 cam.cam_id, consecutive_fails)
                    break
                # Transient drop — do not reconnect yet, just try again
                time.sleep(0.05)
                continue

            consecutive_fails = 0
            # Downscale to 1280×720 before storing — all workers use this space.
            # At 2560×1440, person crops fed to custom model (imgsz=320) are 600×700+
            # px and phones become microscopic. At 1280×720 crops are ~300×350 px
            # and phones are clearly visible at imgsz=320.
            _h0, _w0 = frame.shape[:2]
            proc = cv2.resize(frame, (1280, 720)) if _w0 > 1280 else frame
            cam.state.set_frame(proc)
            cam.record_frame()   # Phase 6: update FPS metric

            # Phase 1: feed hard-negative miner (sampled; checks alert state internally)
            if _hard_neg_miner is not None and _frame_count % 30 == 0:
                snap = cam.state.snapshot()
                has_alert = any(ts.phone_session_start is not None
                                for ts in cam.tracks.values())
                _hard_neg_miner.push_frame(cam.cam_id, frame, has_alert)

            # Write shared frame for web_ui (every 3rd frame ≈ 3fps write rate)
            _frame_count += 1
            if _frame_count % 3 == 0:
                try:
                    shared = cv2.resize(proc, (640, 360))
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
            # session_ptype is authoritative once a session starts;
            # fall back to live type, then pose signal
            ptype = (ts.phone_session_ptype or ts.phone_type or
                     ('ear' if ts.phone_on_ear else 'hand'))
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

def _load_camera_configs() -> List[dict]:
    if os.path.exists("cameras.json"):
        with open("cameras.json") as f:
            cfgs = json.load(f)
        _log.info("Loaded %d cameras from cameras.json", len(cfgs))
        return cfgs
    _log.info("cameras.json not found — using single camera from .env")
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

def _check_mode():
    """
    --check mode: validate all models load and cameras connect without starting
    full detection pipeline.  Exits 0 on success, 1 on failure.
    """
    import sys
    _log.info("=== CHECK MODE ===")
    ok = True

    # Models
    try:
        _load_models()
        _log.info("✓ Models loaded")
    except Exception as exc:
        _log.error("✗ Model load failed: %s", exc); ok = False

    # Cameras
    cfgs = _load_camera_configs()
    for cfg in cfgs:
        cam = CameraSession(cfg)
        url = cam.rtsp_url()
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            _log.error("✗ Cannot open camera %s: %s", cfg['name'], url); ok = False
        else:
            ret, _ = cap.read()
            if not ret:
                _log.warning("⚠  Camera %s opened but no frame", cfg['name'])
            else:
                _log.info("✓ Camera %s — OK", cfg['name'])
        cap.release()

    # Component imports
    _log.info("v4 tracker  : %s", "✓ active" if _TRACKER_V4  else "✗ tracker.py missing")
    _log.info("v4 fusion   : %s", "✓ active" if _FUSION_V4   else "✗ fusion.py missing")
    _log.info("v4 autolabel: %s", "✓ active" if _AUTOLABEL_V4 else "⚠  auto_label.py missing")
    _log.info("v4 mining   : %s", "✓ active" if _MINING_V4   else "⚠  mining.py missing")

    _log.info("=== CHECK %s ===", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Employee Monitor v4")
    ap.add_argument("--check", action="store_true",
                    help="Validate models and cameras without starting detection")
    args, _ = ap.parse_known_args()
    if args.check:
        _check_mode(); return

    global _hard_neg_miner, _desk_reid

    gpu  = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
    vram = (torch.cuda.get_device_properties(0).total_memory // 1024 // 1024
            if DEVICE == "cuda" else 0)

    # ── Phase 6: Graceful degradation on low VRAM ─────────────────────────────
    low_vram = DEVICE == "cuda" and vram < 2048
    if low_vram:
        _log.warning("Low VRAM (%d MB < 2GB) — disabling FaceWorker & OllamaVerifier, "
                     "reducing batch", vram)

    _log.info("═" * 65)
    _log.info("  GPU          : %s  (%d MB)%s", gpu, vram,
              "  [LOW VRAM MODE]" if low_vram else "")
    _log.info("  Tracker      : %s",
              "ByteTrack-Kalman+ReID (v4)" if _TRACKER_V4 else "KalmanTracker (v3)")
    _log.info("  Fusion       : %s",
              "PhoneFusion+SleepFusion (v4)" if _FUSION_V4 else "EvidenceAcc (v3)")
    _log.info("  AutoLabel    : %s", "active" if _AUTOLABEL_V4 else "disabled")
    _log.info("  HardNeg      : %s", "active" if _MINING_V4   else "disabled")
    _log.info("═" * 65)

    _log.info("Loading shared models...")
    _load_models()

    cfgs    = _load_camera_configs()
    cameras = [CameraSession(c) for c in cfgs]
    n       = len(cameras)

    # Phase 6: warn if too many cameras for idle config
    if n > 8:
        _log.warning("%d cameras — enabling aggressive motion-gating to maintain latency", n)

    _log.info("Initializing %d camera(s)...", n)

    # ── Phase 3.2: global desk re-ID registry ─────────────────────────────────
    if _TRACKER_V4:
        _desk_reid = DeskZoneReID()
        _log.info("✓ DeskZoneReID registry active")

    # ── Phase 1: v4 data-collection services ──────────────────────────────────
    if _AUTOLABEL_V4:
        AutoLabelService().start()
        _log.info("✓ AutoLabelService started")

    if _MINING_V4:
        _hard_neg_miner = HardNegativeMiner()
        _hard_neg_miner.start()
        _log.info("✓ HardNegativeMiner started")

    # Start RTSP reader threads — staggered by 1.5s each
    for i, cam in enumerate(cameras):
        delay = i * 1.5
        t = threading.Thread(target=camera_reader, args=(cam, delay), daemon=True)
        t.start()

    # Wait for at least one live frame
    _log.info("Waiting for cameras...")
    for _ in range(150):
        if any(c.state.get_frame() is not None for c in cameras): break
        time.sleep(0.1)
    live = sum(1 for c in cameras if c.state.get_frame() is not None)
    _log.info("%d/%d cameras live.", live, n)

    # Shared batch workers
    w_detect = BatchDetectWorker(cameras)
    w_pose   = BatchPoseWorker(cameras)
    w_detect.start(); w_pose.start()
    _log.info("✓ BatchDetectWorker  (motion-gated, all %d cams)", n)
    _log.info("✓ BatchPoseWorker    (yolov8n-pose, all %d cams)", n)

    # Per-camera workers
    for cam in cameras:
        MotionWorker(cam).start()
        if not low_vram:
            FaceWorker(cam).start()   # disabled in low-VRAM mode
        BehaviorEngine(cam).start()
        _log.info("✓ Workers for %s", cam.name)

    # Ollama false-positive verifier (disabled in low-VRAM mode)
    if not low_vram:
        OllamaVerifier().start()
        _log.info("✓ OllamaVerifier")

    # Periodic DeskZoneReID cleanup
    if _desk_reid is not None:
        def _reid_cleanup():
            while _running:
                time.sleep(60)
                _desk_reid.expire()
        threading.Thread(target=_reid_cleanup, daemon=True, name="ReIDCleanup").start()

    HEADLESS = os.getenv("HEADLESS", "true").lower() != "false"
    _log.info("Employee Monitor v4 — %d cam(s). Headless=%s. Ctrl+C to stop.",
              n, HEADLESS)

    if HEADLESS:
        while _running:
            time.sleep(1)
    else:
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
    _log.info("Stopped.")


if __name__ == "__main__":
    main()
