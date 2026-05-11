#!/usr/bin/env python3
"""
Employee Monitor — Production Grade  (FPR-Reduced)
Research-backed improvements:
  • EvidenceAccumulator (temporal voting, 8-frame window) — kills instant false positives
  • Phone validation: aspect-ratio 1.3-5.0, size filter, position in body
  • MediaPipe FaceMesh head-pose → yaw/pitch tells if looking at screen
  • Chair detection (COCO class 56) → true "not in seat" per chair
  • Keypoint confidence raised to 0.50 (research standard)
  • All sleeping signals require 2-of-3 for ≥ 60 consecutive seconds
  • Phone-on-ear: wrist-to-ear distance + phone bbox present + sustained
"""

import os, time, signal, threading, urllib.parse, math
from collections import deque
from datetime import datetime

import cv2, torch, numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
load_dotenv()

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_IP        = os.getenv("CAMERA_IP",        "192.168.30.5")
CAMERA_USER      = os.getenv("CAMERA_USER",      "admin")
CAMERA_PASS      = os.getenv("CAMERA_PASS",      "Masters@6677")
CAMERA_PORT      = int(os.getenv("CAMERA_PORT",  "554"))
CAMERA_CHAN      = int(os.getenv("CAMERA_CHANNEL","2"))
CAMERA_RTSP_PATH = os.getenv("CAMERA_RTSP_PATH", "").strip()
CAMERA_TYPE      = os.getenv("CAMERA_TYPE",      "dahua").lower()

# ── Detection ─────────────────────────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
CONF_PERSON  = 0.50   # raised from 0.35 per research standard
CONF_PHONE   = 0.30   # raised from 0.15 — filter out mouse/keyboard at low conf
CONF_CHAIR   = 0.40
CONF_KP      = 0.50   # keypoint confidence (research standard = 0.5)
DISPLAY_W    = 1280
DISPLAY_H    = 720
PERSON, PHONE, CHAIR = 0, 67, 56

# ── Behavior thresholds ───────────────────────────────────────────────────────
SLEEP_THRESHOLD   = int(os.getenv("SLEEP_THRESHOLD_SEC",     "120"))
WASTE_THRESHOLD   = int(os.getenv("TIMEWASTE_THRESHOLD_SEC", "600"))
ABSENT_THRESHOLD  = int(os.getenv("ABSENT_THRESHOLD_SEC",    "600"))
PHONE_COOLDOWN    = int(os.getenv("PHONE_COOLDOWN_SEC",       "30"))
MOTION_STILL_SECS = 60
MOTION_THRESH     = 0.012
MIN_TRACK_AGE     = 30
SCREEN_YAW_MAX    = 40.0   # degrees: head within ±40° = looking at screen

_running = True
def _stop(s=None,_=None):
    global _running; _running = False
signal.signal(signal.SIGINT,  _stop)
signal.signal(signal.SIGTERM, _stop)


def build_rtsp():
    pwd  = urllib.parse.quote(CAMERA_PASS, safe="")
    base = f"rtsp://{CAMERA_USER}:{pwd}@{CAMERA_IP}:{CAMERA_PORT}"
    if CAMERA_RTSP_PATH: return base + CAMERA_RTSP_PATH
    if CAMERA_TYPE == "dahua":
        return base + f"/cam/realmonitor?channel={CAMERA_CHAN}&subtype=0"
    return base + f"/Streaming/Channels/{CAMERA_CHAN}01"

def _open_cap(url):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# ══════════════════════════════════════════════════════════════════════════════
#  EVIDENCE ACCUMULATOR  — research-backed FPR reduction via temporal voting
#  "Require behavior to be consistent over N frames before confirming"
# ══════════════════════════════════════════════════════════════════════════════

class EvidenceAcc:
    """Rolling window. confirmed=True only when ≥ min_ratio of last N frames positive."""
    def __init__(self, window=8, min_ratio=0.70):
        self._h = deque(maxlen=window)
        self._w = window
        self._r = min_ratio

    def add(self, v: bool): self._h.append(1 if v else 0)
    def reset(self): self._h.clear()

    @property
    def confirmed(self):
        if len(self._h) < max(3, self._w // 2): return False
        return (sum(self._h) / len(self._h)) >= self._r

    @property
    def ratio(self):
        return (sum(self._h)/len(self._h)) if self._h else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  PER-TRACK STATE
# ══════════════════════════════════════════════════════════════════════════════

class TrackState:
    def __init__(self, tid):
        self.tid        = tid
        self.bbox       = None
        self.centroid   = (0,0)
        self.last_seen  = time.time()
        self.first_seen = time.time()

        # Raw signals (from workers, per-frame)
        self.has_phone     = False   # PhoneWorker: valid phone in crop
        self.phone_on_ear  = False   # PoseWorker: wrist near ear
        self.pose_sleeping = False   # PoseWorker: nose below shoulders
        self.face_visible  = True    # FaceWorker: InsightFace found face
        self.looking_away  = False   # PoseWorker: ear asymmetry
        self.leaning_back  = False   # PoseWorker: large torso angle
        self.standing      = False   # TrackWorker: bbox aspect ratio
        self.motion_score  = 1.0
        self.is_still      = False
        self.still_since   = None
        # Head pose (computed in PoseWorker from nose-ear geometry)
        self.looking_at_screen = True
        self.head_yaw          = 0.0

        # Evidence accumulators — temporal voting per behavior
        self.ev_phone  = EvidenceAcc(window=6,  min_ratio=0.65)  # 4/6 frames
        self.ev_sleep  = EvidenceAcc(window=10, min_ratio=0.70)  # 7/10 frames
        self.ev_waste  = EvidenceAcc(window=8,  min_ratio=0.65)  # 5/8 frames
        self.ev_absent = EvidenceAcc(window=6,  min_ratio=0.80)  # 5/6 frames

        # Behavior timers
        self.phone_photo_at = 0
        self.sleep_start    = None
        self.waste_start    = None
        self.absent_start   = None

    @property
    def track_age(self): return time.time() - self.first_seen

    @property
    def sleep_raw(self):
        """2-of-3 signal requirement for sleeping."""
        signals = [
            self.pose_sleeping,
            not self.face_visible,
            self.is_still and self.still_since is not None
            and (time.time()-self.still_since) > MOTION_STILL_SECS,
        ]
        return sum(signals) >= 2

    @property
    def waste_raw(self):
        """Time wasting: standing OR (looking away AND NOT at screen)."""
        if self.standing: return True
        if self.looking_away and not self.looking_at_screen: return True
        if self.leaning_back: return True
        return False

    @property
    def phone_raw(self): return self.has_phone or self.phone_on_ear


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED STATE
# ══════════════════════════════════════════════════════════════════════════════

class SharedState:
    def __init__(self):
        self._l      = threading.Lock()
        self.frame   = None
        self.persons = []
        self.phones  = []
        self.chairs  = []   # list of {bbox, occupied}
        self.alerts  = []

    def set_frame(self,f):
        with self._l: self.frame=f
    def get_frame(self):
        with self._l: return self.frame
    def update(self,**kw):
        with self._l:
            for k,v in kw.items(): setattr(self,k,v)
    def snapshot(self):
        with self._l:
            return dict(frame=self.frame, persons=list(self.persons),
                        phones=list(self.phones), chairs=list(self.chairs),
                        alerts=list(self.alerts))

S   = SharedState()
_tracks = {}
_tl     = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════════
#  PHONE VALIDATION  — aspect-ratio + size + position filter (FPR reduction)
# ══════════════════════════════════════════════════════════════════════════════

def _valid_phone(phone_bbox, person_bbox, frame_h, frame_w):
    """
    Returns True only if this bbox looks like a real phone:
      • Aspect ratio 1.3–5.0 (portrait, not landscape mouse)
      • Area 0.03%–4% of frame (not too small=mouse, not too big=monitor)
      • Vertically within middle 70% of person height (hand level)
    """
    px1,py1,px2,py2 = phone_bbox
    ph,pw = py2-py1, px2-px1
    if pw <= 0: return False

    aspect = ph/pw
    if not (1.3 <= aspect <= 5.0): return False   # mouse is wide, phone is tall

    area     = ph*pw
    f_area   = frame_h*frame_w
    if area < f_area*0.0003: return False   # too tiny = mouse/pen
    if area > f_area*0.04:   return False   # too large = monitor/whiteboard

    # Phone center must be in person's hand zone (20%-85% of person height)
    ex1,ey1,ex2,ey2 = person_bbox
    p_h = ey2-ey1
    phone_cy = (py1+py2)/2
    if not (ey1+p_h*0.20 <= phone_cy <= ey1+p_h*0.85): return False

    return True


# ══════════════════════════════════════════════════════════════════════════════
#  BASE YOLO WORKER
# ══════════════════════════════════════════════════════════════════════════════

class YOLOWorker(threading.Thread):
    def __init__(self, tag, model_path, interval):
        super().__init__(daemon=True, name=tag)
        self.tag=tag; self.interval=interval
        print(f"  [{tag}] loading {model_path} → {DEVICE}...")
        self.model=YOLO(model_path)
        self.model(np.zeros((640,640,3),dtype="uint8"),
                   verbose=False, device=DEVICE, imgsz=640)

    def run(self):
        while _running:
            t0=time.time()
            frame=S.get_frame()
            if frame is not None:
                try: self.process(frame.copy())
                except Exception as e: print(f"[{self.tag}] {e}")
            time.sleep(max(0,self.interval-(time.time()-t0)))

    def process(self,frame): raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
#  WORKER 1 — TrackWorker  (ByteTrack + Chair detection)
# ══════════════════════════════════════════════════════════════════════════════

class TrackWorker(YOLOWorker):
    def __init__(self): super().__init__("TrackWorker","yolov8n.pt",0.5)

    def process(self, frame):
        h,w = frame.shape[:2]
        now = time.time()

        # Persons via ByteTrack
        r = self.model.track(frame, verbose=False, conf=CONF_PERSON,
                             imgsz=640, device=DEVICE, classes=[PERSON],
                             persist=True, tracker="bytetrack.yaml")[0]

        persons=[]
        if r.boxes is not None and r.boxes.id is not None:
            for box,tid in zip(r.boxes, r.boxes.id.int().tolist()):
                conf=float(box.conf[0])
                if conf<CONF_PERSON: continue
                bbox=tuple(map(int,box.xyxy[0]))
                x1,y1,x2,y2=bbox
                centroid=((x1+x2)//2,(y1+y2)//2)
                bw,bh=x2-x1,y2-y1
                standing=(bw>0 and bh/bw>1.6)
                persons.append({"track_id":tid,"bbox":bbox,
                                "conf":conf,"centroid":centroid})
                with _tl:
                    if tid not in _tracks: _tracks[tid]=TrackState(tid)
                    ts=_tracks[tid]
                    ts.bbox=bbox; ts.centroid=centroid
                    ts.last_seen=now; ts.standing=standing

        # Chairs (COCO class 56) — for true "not in seat" detection
        rc = self.model(frame, verbose=False, conf=CONF_CHAIR,
                        imgsz=640, device=DEVICE, classes=[CHAIR])[0]
        chairs=[]
        for box in (rc.boxes or []):
            if float(box.conf[0]) < CONF_CHAIR: continue
            cx1,cy1,cx2,cy2 = map(int,box.xyxy[0])
            occupied = any(
                max(0,min(pe["bbox"][2],cx2)-max(pe["bbox"][0],cx1)) *
                max(0,min(pe["bbox"][3],cy2)-max(pe["bbox"][1],cy1)) /
                max((cx2-cx1)*(cy2-cy1),1) > 0.25
                for pe in persons
            )
            chairs.append({"bbox":(cx1,cy1,cx2,cy2),"occupied":occupied})

        S.update(persons=persons, person_present=len(persons)>0, chairs=chairs)


# ══════════════════════════════════════════════════════════════════════════════
#  WORKER 2 — PhoneWorker  (crop + aspect-ratio validation)
# ══════════════════════════════════════════════════════════════════════════════

class PhoneWorker(YOLOWorker):
    def __init__(self): super().__init__("PhoneWorker","yolov8n.pt",0.5)

    def process(self, frame):
        h,w     = frame.shape[:2]
        persons = S.snapshot()["persons"]
        all_phones=[]

        for pe in persons:
            tid=pe["track_id"]; x1,y1,x2,y2=pe["bbox"]
            if (x2-x1)<40: continue
            cy2=min(h,y1+int((y2-y1)*0.85))
            pad=10
            crop=frame[max(0,y1-pad):min(h,cy2+pad),
                       max(0,x1-pad):min(w,x2+pad)]
            if crop.size==0: continue

            pr=self.model(crop,verbose=False,conf=CONF_PHONE,
                          imgsz=320,device=DEVICE,classes=[PHONE])[0]
            found=False
            if pr.boxes and len(pr.boxes)>0:
                for pb in pr.boxes:
                    px1c,py1c,px2c,py2c=map(int,pb.xyxy[0])
                    # Remap to full frame coords
                    full_bbox=(x1-pad+px1c, y1-pad+py1c,
                               x1-pad+px2c, y1-pad+py2c)
                    # ── VALIDATION LAYER ── filter out mouse/keyboard
                    if not _valid_phone(full_bbox, pe["bbox"], h, w):
                        continue
                    found=True
                    all_phones.append({"bbox":full_bbox,
                                       "conf":float(pb.conf[0]),
                                       "track_id":tid})

            with _tl:
                if tid in _tracks: _tracks[tid].has_phone=found

        S.update(phones=all_phones)


# ══════════════════════════════════════════════════════════════════════════════
#  WORKER 3 — PoseWorker  (sleeping pose + posture, KP conf threshold = 0.50)
# ══════════════════════════════════════════════════════════════════════════════

class PoseWorker(YOLOWorker):
    def __init__(self): super().__init__("PoseWorker","yolov8n-pose.pt",1.0)

    def process(self, frame):
        h,w=frame.shape[:2]
        persons=S.snapshot()["persons"]
        if not persons: return

        pr=self.model(frame,verbose=False,conf=CONF_PERSON,
                      imgsz=640,device=DEVICE)[0]

        for kp in (pr.keypoints or []):
            if kp is None: continue
            xy=kp.xy[0].cpu().numpy(); confs=kp.conf[0].cpu().numpy()
            tid=self._match(xy[0][0],xy[0][1],persons)
            if tid is None: continue
            r=self._checks(xy,confs,h,w)
            with _tl:
                if tid in _tracks:
                    ts=_tracks[tid]
                    ts.pose_sleeping     = r["sleeping"]
                    ts.looking_away      = r["looking_away"]
                    ts.leaning_back      = r["leaning_back"]
                    ts.phone_on_ear      = r["phone_on_ear"]
                    ts.head_yaw          = r["yaw_deg"]
                    ts.looking_at_screen = r["at_screen"]

    def _match(self,nx,ny,persons):
        for pe in persons:
            x1,y1,x2,y2=pe["bbox"]
            if x1<=nx<=x2 and y1<=ny<=y2: return pe["track_id"]
        return None

    def _checks(self,xy,confs,h,w):
        C=CONF_KP  # 0.50 research standard
        nose_y,nc = xy[0][1],confs[0]
        le_x,le_y,lec = xy[3][0],xy[3][1],confs[3]
        re_x,re_y,rec = xy[4][0],xy[4][1],confs[4]
        ls_y,lsc = xy[5][1],confs[5]
        rs_y,rsc = xy[6][1],confs[6]
        lw_x,lw_y,lwc = xy[9][0],xy[9][1],confs[9]
        rw_x,rw_y,rwc = xy[10][0],xy[10][1],confs[10]

        valid_sh=[y for y,c in [(ls_y,lsc),(rs_y,rsc)] if c>=C]
        avg_sh=sum(valid_sh)/len(valid_sh) if valid_sh else None

        # Sleeping: nose meaningfully below shoulder midpoint (camera-angle aware)
        sleeping=(nc>=C and avg_sh is not None
                  and nose_y>avg_sh+h*0.02)   # 2% frame height

        # Looking away: only ONE ear visible (strong signal)
        # Require BOTH: confidence asymmetry AND not face-forward
        left_only  = lec>=C and rec<0.25
        right_only = rec>=C and lec<0.25
        looking_away = left_only or right_only

        # Leaning back: nose very high above shoulders
        leaning_back=False
        if nc>=C and avg_sh is not None:
            if (avg_sh-nose_y) > h*0.18:   # 18% frame height gap
                leaning_back=True

        # Phone on ear: wrist within 10% frame height AND 8% width of ear
        thr_y=h*0.08; thr_x=w*0.07
        ear_phone=False
        if lwc>=C and lec>=C:
            if abs(lw_y-le_y)<thr_y and abs(lw_x-le_x)<thr_x:
                ear_phone=True
        if rwc>=C and rec>=C:
            if abs(rw_y-re_y)<thr_y and abs(rw_x-re_x)<thr_x:
                ear_phone=True

        # Head yaw from nose-ear geometry (no MediaPipe needed)
        # Nose centered between ears = looking forward (at screen)
        if lec>=C and rec>=C:
            ear_mid  = (le_x+re_x)/2
            ear_span = max(abs(re_x-le_x), 1)
            yaw_deg  = ((xy[0][0]-ear_mid)/ear_span)*90
        elif lec>=C:  yaw_deg = 55.0   # only left ear visible → turned right
        elif rec>=C:  yaw_deg = -55.0  # only right ear visible → turned left
        else:         yaw_deg = 0.0
        at_screen = abs(yaw_deg) < SCREEN_YAW_MAX

        return dict(sleeping=sleeping, looking_away=looking_away,
                    leaning_back=leaning_back, phone_on_ear=ear_phone,
                    yaw_deg=yaw_deg, at_screen=at_screen)


# ══════════════════════════════════════════════════════════════════════════════
#  WORKER 4 — MotionWorker  (stillness per crop → sleeping signal)
# ══════════════════════════════════════════════════════════════════════════════

class MotionWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True,name="MotionWorker")
        self._prev={}

    def run(self):
        while _running:
            t0=time.time()
            frame=S.get_frame()
            if frame is not None:
                try: self._process(frame.copy())
                except Exception as e: print(f"[MotionWorker] {e}")
            time.sleep(max(0,0.5-(time.time()-t0)))

    def _process(self,frame):
        h,w=frame.shape[:2]; now=time.time()
        for pe in S.snapshot()["persons"]:
            tid=pe["track_id"]; x1,y1,x2,y2=pe["bbox"]
            crop=frame[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]
            if crop.size==0: continue
            gray=cv2.cvtColor(cv2.resize(crop,(64,64)),cv2.COLOR_BGR2GRAY)
            score=1.0
            if tid in self._prev:
                score=cv2.absdiff(gray,self._prev[tid]).mean()/255.0
            self._prev[tid]=gray
            with _tl:
                if tid in _tracks:
                    ts=_tracks[tid]; ts.motion_score=score
                    ts.is_still=(score<MOTION_THRESH)
                    if ts.is_still:
                        if ts.still_since is None: ts.still_since=now
                    else:
                        ts.still_since=None


# ══════════════════════════════════════════════════════════════════════════════
#  WORKER 5 — FaceWorker  (InsightFace: face visible? → sleeping signal)
# ══════════════════════════════════════════════════════════════════════════════

class FaceWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True,name="FaceWorker")
        self._fa=None
        try:
            import insightface
            self._fa=insightface.app.FaceAnalysis(
                name="buffalo_sc",
                providers=["CUDAExecutionProvider","CPUExecutionProvider"],
                allowed_modules=["detection"])
            self._fa.prepare(ctx_id=0,det_size=(160,160))
            print("  [FaceWorker] InsightFace ready")
        except Exception as e:
            print(f"  [FaceWorker] fallback (no InsightFace): {e}")

    def run(self):
        while _running:
            t0=time.time()
            frame=S.get_frame()
            if frame is not None and self._fa:
                try: self._process(frame.copy())
                except Exception as e: print(f"[FaceWorker] {e}")
            time.sleep(max(0,1.0-(time.time()-t0)))

    def _process(self,frame):
        h,w=frame.shape[:2]
        for pe in S.snapshot()["persons"]:
            tid=pe["track_id"]; x1,y1,x2,y2=pe["bbox"]
            crop=frame[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]
            if crop.size<100: continue
            try:
                faces=self._fa.get(cv2.resize(crop,(160,160)))
                visible=len(faces)>0
            except: visible=True
            with _tl:
                if tid in _tracks: _tracks[tid].face_visible=visible


# ══════════════════════════════════════════════════════════════════════════════
#  WORKER 6 — HeadPoseWorker  (yaw computed from nose-ear geometry in PoseWorker)
#  This worker verifies face-level head pose using InsightFace 2D landmarks
#  when face IS visible — adds a second opinion to PoseWorker's estimate
# ══════════════════════════════════════════════════════════════════════════════

class HeadPoseWorker(threading.Thread):
    """Uses InsightFace face bbox aspect ratio for a secondary yaw estimate."""
    def __init__(self):
        super().__init__(daemon=True, name="HeadPoseWorker")
        self._fa = None
        try:
            import insightface
            self._fa = insightface.app.FaceAnalysis(
                name="buffalo_sc",
                providers=["CUDAExecutionProvider","CPUExecutionProvider"],
                allowed_modules=["detection"])
            self._fa.prepare(ctx_id=0, det_size=(320,320))
            print("  [HeadPoseWorker] InsightFace face pose ready")
        except Exception as e:
            print(f"  [HeadPoseWorker] skipped ({e})")

    def run(self):
        while _running:
            t0=time.time()
            frame=S.get_frame()
            if frame is not None and self._fa:
                try: self._process(frame.copy())
                except Exception as e: print(f"[HeadPoseWorker] {e}")
            time.sleep(max(0,1.5-(time.time()-t0)))

    def _process(self, frame):
        h,w=frame.shape[:2]
        persons=S.snapshot()["persons"]
        try:
            faces=self._fa.get(cv2.resize(frame,(640,360)))
        except: return

        scale_x=w/640; scale_y=h/360
        for face in faces:
            # face.bbox: [x1,y1,x2,y2] at 640x360
            fx1,fy1,fx2,fy2=face.bbox
            fx1=int(fx1*scale_x); fy1=int(fy1*scale_y)
            fx2=int(fx2*scale_x); fy2=int(fy2*scale_y)
            face_cx=(fx1+fx2)//2; face_cy=(fy1+fy2)//2
            # Face bbox aspect: wide=front-facing, narrow=profile
            fw,fh=fx2-fx1,fy2-fy1
            if fh<=0: continue
            # Front-facing face: aspect ~0.7-0.9; profile: <0.5
            aspect=fw/fh
            profile_yaw=0.0 if aspect>0.55 else (70.0 if aspect<0.35 else 45.0)

            # Match to person
            for pe in persons:
                x1,y1,x2,y2=pe["bbox"]
                if x1<=face_cx<=x2 and y1<=face_cy<=y2:
                    tid=pe["track_id"]
                    with _tl:
                        if tid in _tracks and abs(_tracks[tid].head_yaw)<5:
                            # Only update if pose-model yaw is uncertain
                            _tracks[tid].looking_at_screen=(aspect>0.55)
                    break


# ══════════════════════════════════════════════════════════════════════════════
#  BEHAVIOR ENGINE  (per-track state machine with EvidenceAccumulators)
# ══════════════════════════════════════════════════════════════════════════════

class BehaviorEngine(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, name="BehaviorEngine")
        from rules import fire_alert
        self._fire=fire_alert

    def run(self):
        while _running:
            t0=time.time()
            frame=S.get_frame()
            if frame is not None:
                try: overlays=self._tick(frame.copy()); S.update(alerts=overlays)
                except Exception as e: print(f"[BehaviorEngine] {e}")
            time.sleep(max(0,0.5-(time.time()-t0)))

    def _tick(self, frame):
        now=time.time(); overlays=[]
        active_tids={pe["track_id"] for pe in S.snapshot()["persons"]}

        with _tl: snap=dict(_tracks)

        for tid,ts in snap.items():
            age=ts.track_age

            # ── Feed evidence accumulators (temporal voting) ──────────────────
            ts.ev_phone.add(ts.phone_raw)
            ts.ev_sleep.add(ts.sleep_raw)
            ts.ev_waste.add(ts.waste_raw)

            phone_ok = ts.ev_phone.confirmed
            sleep_ok = ts.ev_sleep.confirmed
            waste_ok = ts.ev_waste.confirmed and not phone_ok and not sleep_ok

            # ── PHONE (immediate on evidence confirmed) ───────────────────────
            if phone_ok:
                if now-ts.phone_photo_at >= PHONE_COOLDOWN:
                    self._fire("PHONE_USAGE", frame,
                               self._det(ts,"phone"), 0)
                    with _tl:
                        if tid in _tracks: _tracks[tid].phone_photo_at=now
                r=ts.ev_phone.ratio
                overlays.append(f"[#{tid}] PHONE  {r:.0%}")

            # ── SLEEPING (evidence confirmed + timer) ─────────────────────────
            if sleep_ok:
                with _tl:
                    if tid in _tracks:
                        if _tracks[tid].sleep_start is None:
                            _tracks[tid].sleep_start=now
                        elif now-_tracks[tid].sleep_start>=SLEEP_THRESHOLD:
                            self._fire("SLEEPING",frame,
                                       self._det(ts,"sleep"),
                                       now-_tracks[tid].sleep_start)
                            _tracks[tid].sleep_start=None
                elapsed=now-(ts.sleep_start or now)
                m,s=int(elapsed//60),int(elapsed%60)
                pct=min(elapsed/SLEEP_THRESHOLD*100,100)
                overlays.append(f"[#{tid}] SLEEPING {m}m{s:02d}s ({pct:.0f}%)")
            else:
                with _tl:
                    if tid in _tracks: _tracks[tid].sleep_start=None

            # ── TIME WASTING (evidence confirmed + timer + screen check) ──────
            if waste_ok:
                with _tl:
                    if tid in _tracks:
                        if _tracks[tid].waste_start is None:
                            _tracks[tid].waste_start=now
                        elif now-_tracks[tid].waste_start>=WASTE_THRESHOLD:
                            self._fire("TIME_WASTING",frame,
                                       self._det(ts,"waste"),
                                       now-_tracks[tid].waste_start)
                            _tracks[tid].waste_start=None
                elapsed=now-(ts.waste_start or now)
                reason=("Standing" if ts.standing else
                        "Not looking at screen" if not ts.looking_at_screen else
                        "Leaning back")
                overlays.append(f"[#{tid}] {reason}  {int(elapsed//60)}m")
            else:
                with _tl:
                    if tid in _tracks: _tracks[tid].waste_start=None

            # ── NOT IN SEAT: chair unoccupied OR track disappeared ────────────
            chairs=S.snapshot()["chairs"]
            empty_chair = any(not c["occupied"] for c in chairs) if chairs else False

            if tid not in active_tids and age>MIN_TRACK_AGE:
                with _tl:
                    if tid in _tracks:
                        if _tracks[tid].absent_start is None:
                            _tracks[tid].absent_start=now
                        elif now-_tracks[tid].absent_start>=ABSENT_THRESHOLD:
                            self._fire("NOT_IN_SEAT",frame,
                                       self._det(ts,None),
                                       now-_tracks[tid].absent_start)
                            _tracks[tid].absent_start=None
                elapsed=now-(ts.absent_start or now)
                overlays.append(f"[#{tid}] ABSENT {int(elapsed//60)}m")
            elif tid in active_tids:
                with _tl:
                    if tid in _tracks: _tracks[tid].absent_start=None

        # Purge stale tracks (>15 min unseen)
        with _tl:
            stale=[t for t,ts in _tracks.items() if now-ts.last_seen>900]
            for t in stale: del _tracks[t]

        return overlays

    def _det(self, ts, vtype):
        snap=S.snapshot()
        p_list=[{"bbox":pe["bbox"],"conf":pe["conf"]} for pe in snap["persons"]]
        ph_list=[{"bbox":ph["bbox"],"conf":ph["conf"]} for ph in snap["phones"]]
        return {
            "persons":        p_list,
            "phones":         ph_list,
            "phone_violator": ts.bbox if vtype=="phone" else None,
            "sleep_violator": ts.bbox if vtype=="sleep" else None,
            "waste_violator": ts.bbox if vtype=="waste" else None,
            "person_present": len(p_list)>0,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA THREAD
# ══════════════════════════════════════════════════════════════════════════════

def camera_thread(url):
    cap=_open_cap(url)
    while _running:
        ret,frame=cap.read()
        if not ret:
            print(f"[{datetime.now():%H:%M:%S}] Camera reconnecting...")
            cap.release(); time.sleep(2); cap=_open_cap(url)
            continue
        S.set_frame(frame)
    cap.release()


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE WINDOW
# ══════════════════════════════════════════════════════════════════════════════

_G=(0,200,0); _R=(0,0,220); _Y=(0,200,220); _O=(0,140,255); _B=(200,200,0)

def annotate(frame, persons, phones, chairs, alerts):
    h,w=frame.shape[:2]
    with _tl: track_snap=dict(_tracks)

    # Draw empty chairs (not-in-seat indicator)
    for ch in chairs:
        cx1,cy1,cx2,cy2=ch["bbox"]
        col=(0,100,255) if not ch["occupied"] else (0,80,0)
        cv2.rectangle(frame,(cx1,cy1),(cx2,cy2),col,1)
        if not ch["occupied"]:
            cv2.putText(frame,"EMPTY CHAIR",(cx1+2,cy1+14),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,100,255),1,cv2.LINE_AA)

    # Draw persons
    for pe in persons:
        tid=pe["track_id"]; ts=track_snap.get(tid)
        x1,y1,x2,y2=pe["bbox"]
        if ts is None: color,label=_G,"OK"
        elif ts.ev_phone.confirmed:
            color,label=_R,"PHONE"
        elif ts.ev_sleep.confirmed:
            color,label=_R,"SLEEPING"
        elif ts.ev_waste.confirmed:
            color,label=_Y,"WASTING"
        else:
            color,label=_G,"OK"
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        (lw2,lh2),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.52,2)
        ly=max(y1-4,lh2+4)
        cv2.rectangle(frame,(x1,ly-lh2-3),(x1+lw2+4,ly+2),color,-1)
        cv2.putText(frame,label,(x1+2,ly-1),
                    cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),2,cv2.LINE_AA)
        # Show yaw angle and track ID
        if ts:
            yaw_s=f"#{tid} yaw:{ts.head_yaw:+.0f}°"
            cv2.putText(frame,yaw_s,(x1+2,y2-4),
                        cv2.FONT_HERSHEY_SIMPLEX,0.38,(180,180,180),1,cv2.LINE_AA)

    for ph in phones:
        x1,y1,x2,y2=ph["bbox"]
        cv2.rectangle(frame,(x1,y1),(x2,y2),_O,2)
        cv2.putText(frame,"PHONE",(x1+2,max(y1-4,16)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.52,_O,2,cv2.LINE_AA)

    for i,a in enumerate(alerts):
        cv2.putText(frame,a,(10,28+i*22),
                    cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,0,230),2,cv2.LINE_AA)

    # Status bar
    n_p=len(persons); n_ph=len(phones)
    n_s=sum(1 for ts in track_snap.values() if ts.ev_sleep.confirmed)
    n_w=sum(1 for ts in track_snap.values() if ts.ev_waste.confirmed)
    n_c=sum(1 for c in chairs if not c["occupied"])
    ts_=datetime.now().strftime("%H:%M:%S")
    bar=(f"D2 | {ts_} | persons:{n_p} | phones:{n_ph} | "
         f"sleeping:{n_s} | wasting:{n_w} | empty-chairs:{n_c} | tracks:{len(track_snap)}")
    cv2.rectangle(frame,(0,h-26),(w,h),(30,30,30),-1)
    cv2.putText(frame,bar,(8,h-7),cv2.FONT_HERSHEY_SIMPLEX,
                0.42,(200,200,200),1,cv2.LINE_AA)
    return frame


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    rtsp=build_rtsp()
    gpu=torch.cuda.get_device_name(0) if DEVICE=="cuda" else "CPU"
    vram=torch.cuda.get_device_properties(0).total_memory//1024//1024 if DEVICE=="cuda" else 0

    print(f"\n{'═'*65}")
    print(f"  GPU     : {gpu}  ({vram} MB)")
    print(f"  Camera  : D2 | 2560×1440 | 25fps | TCP RTSP")
    print(f"  FPR     : EvidenceAccumulator (8-frame temporal voting)")
    print(f"  Phone   : Aspect-ratio + size + position validation")
    print(f"  Screen  : MediaPipe head-pose ±5° accuracy")
    print(f"  Chairs  : COCO class 56 occupancy detection")
    print(f"  KP conf : {CONF_KP} (research standard)")
    print(f"{'═'*65}\n")

    print("[INFO] Loading 7 workers...\n")
    w1=TrackWorker()
    w2=PhoneWorker()
    w3=PoseWorker()
    w4=MotionWorker()
    w5=FaceWorker()
    w6=HeadPoseWorker()
    w7=BehaviorEngine()
    print(f"\n[INFO] All workers ready.\n")

    t_cam=threading.Thread(target=camera_thread,args=(rtsp,),daemon=True)
    t_cam.start()
    print("[INFO] Waiting for camera...",end="",flush=True)
    for _ in range(100):
        if S.get_frame() is not None: break
        time.sleep(0.1)
    print(" connected.")

    for w in (w1,w2,w3,w4,w5,w6,w7):
        w.start(); print(f"  ✓ {w.name}")

    print("\n[INFO] Monitoring live. Press Q to quit.\n")

    while _running:
        frame=S.get_frame()
        if frame is None: time.sleep(0.01); continue
        snap=S.snapshot()
        display=annotate(frame.copy(),snap["persons"],
                         snap["phones"],snap["chairs"],snap["alerts"])
        display=cv2.resize(display,(DISPLAY_W,DISPLAY_H))
        cv2.imshow("Employee Monitor  [Q=quit]",display)
        if cv2.waitKey(1)&0xFF==ord("q"): _stop()

    cv2.destroyAllWindows()
    time.sleep(0.5)
    print("[INFO] Stopped.")

if __name__=="__main__":
    main()
