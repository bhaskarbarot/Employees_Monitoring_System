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

import cv2, torch, numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# ── Detection config ──────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
HALF        = DEVICE == "cuda"
CONF_PERSON = 0.50
CONF_PHONE  = 0.27          # slightly lower than v1 for better recall
CONF_CHAIR  = 0.40
CONF_KP     = 0.50
DISPLAY_W   = 1280
DISPLAY_H   = 720
PERSON, PHONE, CHAIR = 0, 67, 56

SLEEP_THRESHOLD   = int(os.getenv("SLEEP_THRESHOLD_SEC",     "120"))
WASTE_THRESHOLD   = int(os.getenv("TIMEWASTE_THRESHOLD_SEC", "600"))
PHONE_COOLDOWN    = int(os.getenv("PHONE_COOLDOWN_SEC",       "30"))
MOTION_STILL_SECS = 60
MOTION_THRESH     = 0.012
MIN_TRACK_AGE     = 30

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
_detect_model = None   # yolov8s.pt — person + phone + chair
_pose_model   = None   # yolov8n-pose.pt
_infer_lock   = threading.Lock()  # serialize GPU calls

def _load_models():
    global _detect_model, _pose_model
    dummy = np.zeros((416, 416, 3), dtype="uint8")
    print(f"  [Models] yolov8s.pt → {DEVICE}  fp16={HALF}")
    _detect_model = YOLO("yolov8s.pt")
    _detect_model([dummy], verbose=False, device=DEVICE, imgsz=416, half=HALF)
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
        self.looking_away = False; self.leaning_back = False
        self.standing = False; self.motion_score = 1.0
        self.is_still = False; self.still_since = None
        self.looking_at_screen = True; self.head_yaw = 0.0
        # phone: confirm after 2 consecutive detections (~1 sec) → immediate capture
        self.ev_phone  = EvidenceAcc(window=4,  min_ratio=0.50, min_frames=2)
        # sleep/waste: need sustained detection (several seconds) before alerting
        self.ev_sleep  = EvidenceAcc(window=10, min_ratio=0.70)
        self.ev_waste  = EvidenceAcc(window=8,  min_ratio=0.65)
        self.phone_photo_at = 0; self.sleep_start = None
        self.waste_start = None

    @property
    def track_age(self): return time.time() - self.first_seen

    @property
    def sleep_raw(self):
        sigs = [
            self.pose_sleeping,
            not self.face_visible,
            self.is_still and self.still_since is not None
            and (time.time() - self.still_since) > MOTION_STILL_SECS,
        ]
        return sum(sigs) >= 2

    @property
    def waste_raw(self):
        return (self.standing
                or (self.looking_away and not self.looking_at_screen)
                or self.leaning_back)

    @property
    def phone_raw(self):
        # Only 'hand' / 'ear' trigger the alert accumulator — desk phones are passive
        return (self.phone_type in ('hand', 'ear')) or self.phone_on_ear


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

def _classify_phone(phone_bbox, person_bbox, phone_on_ear_signal=False):
    """
    Classify where the phone is relative to the person.
    Returns: 'ear' | 'hand' | 'desk'

    Zone map (fraction of person bbox height from top):
      0%  – 28%  → head / ear level  → PHONE ON EAR
      28% – 78%  → shoulder / hand   → PHONE IN HAND
      78% – 100% → lap / desk        → PHONE ON DESK
    PoseWorker phone_on_ear overrides to 'ear'.
    """
    if phone_on_ear_signal:
        return 'ear'
    ex1, ey1, ex2, ey2 = person_bbox
    p_h = max(ey2 - ey1, 1)
    phone_cy = (phone_bbox[1] + phone_bbox[3]) / 2
    rel_y = (phone_cy - ey1) / p_h
    if rel_y < 0.28:   return 'ear'
    if rel_y < 0.78:   return 'hand'
    return 'desk'


def _valid_phone(phone_bbox, person_bbox, frame_h, frame_w):
    px1, py1, px2, py2 = phone_bbox
    ph, pw = py2 - py1, px2 - px1
    if pw <= 0 or ph <= 0: return False
    aspect = ph / pw

    # Portrait (tall) OR landscape (wide) — v1 rejected landscape phones
    portrait  = 1.2 <= aspect <= 5.0
    landscape = 0.20 <= aspect <= 0.83
    if not (portrait or landscape): return False

    area = ph * pw; f_area = frame_h * frame_w
    if area < f_area * 0.0002: return False   # too small
    if area > f_area * 0.05:   return False   # too large (monitor/whiteboard)

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
        ch = c.get('channel', 1)
        if c.get('type', 'dahua').lower() == 'dahua':
            return base + f"/cam/realmonitor?channel={ch}&subtype=0"
        return base + f"/Streaming/Channels/{ch}01"


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
        # Crops are zoomed-in person regions → phone appears 4× larger than
        # it would in the full frame, dramatically improving recall.
        crops = [m[2] for m in crop_meta]
        with _infer_lock:
            r2 = _detect_model(crops, verbose=False, conf=CONF_PHONE,
                               imgsz=320, device=DEVICE, half=HALF,
                               classes=[PHONE])

        phones_by_idx     = [[] for _ in cams]
        phone_tids_by_idx = [set() for _ in cams]  # only hand/ear tids

        for (cam_idx, pe, crop, ox, oy, fh, fw), pr in zip(crop_meta, r2):
            cam = cams[cam_idx]
            if pr.boxes is None: continue
            for pb in pr.boxes:
                pconf = float(pb.conf[0])
                if pconf < CONF_PHONE: continue
                px1, py1, px2, py2 = map(int, pb.xyxy[0])
                full_bbox = (ox+px1, oy+py1, ox+px2, oy+py2)
                if not _valid_phone(full_bbox, pe['bbox'], fh, fw): continue

                # Read PoseWorker's phone_on_ear signal from track state
                ear_signal = False
                with cam.tracks_lock:
                    if pe['track_id'] in cam.tracks:
                        ear_signal = cam.tracks[pe['track_id']].phone_on_ear

                ptype = _classify_phone(full_bbox, pe['bbox'], ear_signal)

                phones_by_idx[cam_idx].append({
                    'bbox': full_bbox, 'conf': pconf,
                    'track_id': pe['track_id'], 'type': ptype
                })

                # Update track state immediately so BehaviorEngine sees it
                with cam.tracks_lock:
                    if pe['track_id'] in cam.tracks:
                        ts = cam.tracks[pe['track_id']]
                        ts.phone_bbox = full_bbox
                        ts.phone_type = ptype
                        ts.has_phone  = ptype in ('hand', 'ear')

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
                    ts.pose_sleeping     = res['sleeping']
                    ts.looking_away      = res['looking_away']
                    ts.leaning_back      = res['leaning_back']
                    ts.phone_on_ear      = res['phone_on_ear']
                    ts.head_yaw          = res['yaw_deg']
                    ts.looking_at_screen = res['at_screen']

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

        sleeping     = (nc >= C and avg_sh is not None
                        and nose_y > avg_sh + sh_span * 0.4)

        # ── Head-turn / looking-away logic — depends on camera angle ─────────
        #
        # side camera  (your camera):
        #   Normal posture = person's profile faces camera → 1 ear visible
        #   Looking away   = person turns toward camera    → BOTH ears visible
        #
        # front / overhead camera:
        #   Normal posture = person faces camera           → both ears visible
        #   Looking away   = person turns sideways         → only 1 ear visible
        #
        left_visible  = lec >= EAR_CONF_VISIBLE
        right_visible = rec >= EAR_CONF_VISIBLE
        both_visible  = left_visible and right_visible
        one_visible   = left_visible != right_visible   # XOR — exactly one

        if CAMERA_ANGLE == 'side':
            looking_away = both_visible          # side cam: both ears = turned away from screen
            at_screen    = one_visible           # side cam: profile = looking at screen
        else:
            looking_away = one_visible           # front cam: one ear = turned away
            at_screen    = both_visible          # front cam: both ears = looking at screen

        leaning_back = (nc >= C and avg_sh is not None
                        and (avg_sh - nose_y) > sh_span * LEANING_BACK_RATIO)

        # Phone on ear: wrist within proportional distance of ear
        ear_phone = False
        thr_y = sh_span * 0.55; thr_x = sh_span * 0.65
        if lwc >= C and lec >= C:
            if abs(lw_y - le_y) < thr_y and abs(lw_x - le_x) < thr_x:
                ear_phone = True
        if rwc >= C and rec >= C:
            if abs(rw_y - re_y) < thr_y and abs(rw_x - re_x) < thr_x:
                ear_phone = True

        # Head yaw (kept for display info)
        if left_visible and right_visible:
            ear_mid  = (le_x + re_x) / 2
            ear_span = max(abs(re_x - le_x), 1)
            yaw_deg  = ((nose_x - ear_mid) / ear_span) * 90
        elif left_visible:  yaw_deg =  55.0
        elif right_visible: yaw_deg = -55.0
        else:               yaw_deg =   0.0

        return dict(sleeping=sleeping, looking_away=looking_away,
                    leaning_back=leaning_back, phone_on_ear=ear_phone,
                    yaw_deg=yaw_deg, at_screen=at_screen)


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
#  BEHAVIOR ENGINE  (per-camera state machine)
# ══════════════════════════════════════════════════════════════════════════════

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

        for tid, ts in snap.items():
            ts.ev_phone.add(ts.phone_raw)
            ts.ev_sleep.add(ts.sleep_raw)
            ts.ev_waste.add(ts.waste_raw)
            phone_ok = ts.ev_phone.confirmed
            sleep_ok = ts.ev_sleep.confirmed
            waste_ok = ts.ev_waste.confirmed and not phone_ok and not sleep_ok

            # ── PHONE  (hand / ear only — desk is display-only, no alert) ───────
            # Captures immediately on first confirmed detection (~1 sec / 2 frames).
            # After capture, waits PHONE_COOLDOWN before next photo.
            # When phone disappears (ev clears), resets so next pickup fires instantly.
            if phone_ok:
                ptype = ts.phone_type or ('ear' if ts.phone_on_ear else 'hand')
                event = "PHONE_EAR" if ptype == 'ear' else "PHONE_HAND"
                if now - ts.phone_photo_at >= PHONE_COOLDOWN:
                    self._fire(event, frame, self._det(ts, "phone"), 0,
                               self.cam.cam_id, self.cam.name)
                    with self.cam.tracks_lock:
                        if tid in self.cam.tracks:
                            self.cam.tracks[tid].phone_photo_at = now
                label = "PHONE ON EAR" if ptype == 'ear' else "PHONE IN HAND"
                overlays.append(f"[{self.cam.name}#{tid}] {label} {ts.ev_phone.ratio:.0%}")
            else:
                # Phone gone — reset cooldown timer so next pickup captures immediately
                if ts.ev_phone.ratio == 0.0 and ts.phone_photo_at != 0:
                    with self.cam.tracks_lock:
                        if tid in self.cam.tracks:
                            self.cam.tracks[tid].phone_photo_at = 0

            # ── SLEEPING ────────────────────────────────────────────────────────
            if sleep_ok:
                with self.cam.tracks_lock:
                    if tid in self.cam.tracks:
                        if self.cam.tracks[tid].sleep_start is None:
                            self.cam.tracks[tid].sleep_start = now
                        elif now - self.cam.tracks[tid].sleep_start >= SLEEP_THRESHOLD:
                            self._fire("SLEEPING", frame, self._det(ts, "sleep"),
                                       now - self.cam.tracks[tid].sleep_start,
                                       self.cam.cam_id, self.cam.name)
                            self.cam.tracks[tid].sleep_start = None
                elapsed = now - (ts.sleep_start or now)
                overlays.append(f"[{self.cam.name}#{tid}] SLEEPING "
                                 f"{int(elapsed//60)}m{int(elapsed%60):02d}s "
                                 f"({min(elapsed/SLEEP_THRESHOLD*100,100):.0f}%)")
            else:
                with self.cam.tracks_lock:
                    if tid in self.cam.tracks: self.cam.tracks[tid].sleep_start = None

            # ── TIME WASTING ────────────────────────────────────────────────────
            if waste_ok:
                with self.cam.tracks_lock:
                    if tid in self.cam.tracks:
                        if self.cam.tracks[tid].waste_start is None:
                            self.cam.tracks[tid].waste_start = now
                        elif now - self.cam.tracks[tid].waste_start >= WASTE_THRESHOLD:
                            self._fire("TIME_WASTING", frame, self._det(ts, "waste"),
                                       now - self.cam.tracks[tid].waste_start,
                                       self.cam.cam_id, self.cam.name)
                            self.cam.tracks[tid].waste_start = None
                elapsed = now - (ts.waste_start or now)
                reason = ("Standing" if ts.standing
                          else "Not at screen" if not ts.looking_at_screen
                          else "Leaning back")
                overlays.append(f"[{self.cam.name}#{tid}] {reason} {int(elapsed//60)}m")
            else:
                with self.cam.tracks_lock:
                    if tid in self.cam.tracks: self.cam.tracks[tid].waste_start = None


        return overlays

    def _det(self, ts, vtype):
        snap = self.cam.state.snapshot()
        return {
            'persons':        [{'bbox': pe['bbox'], 'conf': pe['conf']} for pe in snap['persons']],
            'phones':         [{'bbox': ph['bbox'], 'conf': ph['conf']} for ph in snap['phones']],
            'phone_violator': ts.bbox if vtype == 'phone' else None,
            'sleep_violator': ts.bbox if vtype == 'sleep' else None,
            'waste_violator': ts.bbox if vtype == 'waste' else None,
            'person_present': len(snap['persons']) > 0,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA READER THREAD
# ══════════════════════════════════════════════════════════════════════════════

def camera_reader(cam):
    url = cam.rtsp_url()
    print(f"  [Cam-{cam.cam_id}] {cam.name}  →  {url}")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while _running:
        ret, frame = cap.read()
        if not ret:
            print(f"  [Cam-{cam.cam_id}] Reconnecting...")
            cap.release(); time.sleep(2)
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue
        cam.state.set_frame(frame)
    cap.release()


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

    for pe in snap['persons']:
        tid = pe['track_id']; ts = track_snap.get(tid)
        x1,y1,x2,y2 = pe['bbox']
        if   ts is None:              color, label = _G, "OK"
        elif ts.ev_phone.confirmed:
            ptype = ts.phone_type or ('ear' if ts.phone_on_ear else 'hand')
            if ptype == 'ear': color, label = (0,60,255), "PHONE ON EAR"
            else:              color, label = _R,          "PHONE IN HAND"
        elif ts.ev_sleep.confirmed:   color, label = _R, "SLEEPING"
        elif ts.ev_waste.confirmed:   color, label = _Y, "WASTING"
        else:                         color, label = _G, "OK"
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        (lw2,lh2),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        ly = max(y1-4, lh2+4)
        cv2.rectangle(frame, (x1, ly-lh2-3), (x1+lw2+4, ly+2), color, -1)
        cv2.putText(frame, label, (x1+2, ly-1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 2, cv2.LINE_AA)
        if ts:
            cv2.putText(frame, f"#{tid} {ts.head_yaw:+.0f}°", (x1+2, y2-4),
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
    n_w  = sum(1 for ts in track_snap.values() if ts.ev_waste.confirmed)
    bar  = (f"{cam.name} | {datetime.now():%H:%M:%S} | "
            f"P:{n_p}  Phone:{n_ph}  Sleep:{n_s}  Waste:{n_w}")
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

    # Start RTSP reader threads
    for cam in cameras:
        t = threading.Thread(target=camera_reader, args=(cam,), daemon=True)
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

    print(f"\n[INFO] Monitoring {n} camera(s). Press Q to quit.\n")

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
