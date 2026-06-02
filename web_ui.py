#!/usr/bin/env python3
"""
web_ui.py — Employee Monitor Live Dashboard

Usage:
    python3 web_ui.py          → http://localhost:5000

Features:
    • Live MJPEG stream from every camera
    • Click any camera → enlarged modal + camera-only alerts
    • Alert photos with duration (how long person was using phone / sleeping)
    • Auto-refresh every 5 sec
"""

import os, cv2, json, time, threading, urllib.parse, numpy as np
from pathlib import Path
from datetime import datetime
from flask import Flask, Response, jsonify, send_from_directory

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

app        = Flask(__name__)
app.config["PROPAGATE_EXCEPTIONS"] = True
_BASE      = Path(__file__).parent.resolve()     # always project root, regardless of CWD
PHOTOS_DIR = _BASE / "logs" / "photos"
LOGS_FILE  = _BASE / "logs" / "logs.txt"
PHOTOS_DIR.mkdir(parents=True, exist_ok=True)   # ensure dir exists so /photos/* never 404s

# ── API response cache — avoids re-scanning files on every browser poll ───────
_api_cache     = {}          # key → (data, timestamp)
_cache_lock    = threading.Lock()

def _cached(key, ttl_sec, producer):
    """Return cached result if fresh, else call producer() and cache it."""
    now = time.time()
    with _cache_lock:
        entry = _api_cache.get(key)
        if entry and (now - entry[1]) < ttl_sec:
            return entry[0]
    data = producer()
    with _cache_lock:
        _api_cache[key] = (data, time.time())
    return data

def _bust(key):
    with _cache_lock:
        _api_cache.pop(key, None)


# ── Camera frames — read from monitor.py's shared files (NO direct RTSP) ──────
#
# Architecture: monitor.py holds ALL RTSP connections (7 total).
#   It writes latest frame to /tmp/monitor_frames/{name}.jpg every ~0.3s.
#   web_ui.py reads those files — ZERO additional RTSP connections to NVR.
#   This leaves the attendance system's cameras completely undisturbed.

SHARED_FRAMES_DIR = Path("/tmp/monitor_frames")

class CameraStream:
    """Reads frames from monitor.py's shared files — no RTSP connection."""

    def __init__(self, cam):
        self.id        = cam["id"]
        self.name      = cam["name"]
        self.connected = False
        self._file     = SHARED_FRAMES_DIR / f"{self.name}.jpg"
        self._lock     = threading.Lock()
        self._frame    = None
        threading.Thread(target=self._poll, daemon=True).start()

    def _poll(self):
        """Poll shared frame file written by monitor.py."""
        last_mtime = 0
        STALE_SEC = 5.0   # no update in 5s = monitor stopped → show as disconnected
        while True:
            try:
                if self._file.exists():
                    mt  = self._file.stat().st_mtime
                    now = time.time()
                    if mt != last_mtime:
                        data = np.frombuffer(self._file.read_bytes(), dtype=np.uint8)
                        f    = cv2.imdecode(data, cv2.IMREAD_COLOR)
                        if f is not None:
                            with self._lock:
                                self._frame = f
                            self.connected = True
                            last_mtime = mt
                    else:
                        # File exists but hasn't changed — stale if too old
                        self.connected = (now - mt) < STALE_SEC
                else:
                    self.connected = False
            except Exception:
                pass
            time.sleep(0.15)   # check for new frame every 150ms (~6fps display)

    def jpeg(self, w=640, h=360, q=70):
        with self._lock: f = self._frame
        if f is None:
            f = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.rectangle(f, (0,0), (w,h), (18,18,18), -1)
            cv2.putText(f, self.name, (w//2-30, h//2-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (55,55,55), 2)
            cv2.putText(f, "Waiting for monitor...", (w//2-90, h//2+22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40,40,40), 1)
        else:
            fh, fw = f.shape[:2]
            if fw != w or fh != h:
                f = cv2.resize(f, (w, h))
        _, enc = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, q])
        return enc.tobytes()


# ── Load cameras ──────────────────────────────────────────────────────────────

CAMS       = json.load(open(_BASE / "cameras.json")) if (_BASE / "cameras.json").exists() else []
STREAMS    = {c["name"]: CameraStream(c) for c in CAMS}
CAM_ID_MAP = {c["name"]: f"Cam{c['id']}" for c in CAMS}


# ── Alert parsing helpers ──────────────────────────────────────────────────────

def _parse_photo(p: Path):
    parts = p.stem.split("_")
    try:
        di = next(i for i, v in enumerate(parts) if v.isdigit() and len(v) == 8)
        date_s, time_s = parts[di], parts[di + 1]
        # Parse actual timestamp FROM FILENAME (accurate even after file copy)
        ts = datetime.strptime(f"{date_s}_{time_s}", "%Y%m%d_%H%M%S").timestamp()
        return {
            "cam":   parts[0],
            "event": " ".join(parts[1:di]),
            "date":  date_s,
            "time":  f"{time_s[:2]}:{time_s[2:4]}:{time_s[4:6]}",
            "url":   f"/photos/{p.name}",
            "mtime": ts,   # from filename, not file system
        }
    except Exception:
        return None


def _sessions(photos: list, gap_min=4):
    """Group consecutive alerts of same cam+event into sessions with durations."""
    items = sorted([x for x in [_parse_photo(p) for p in photos] if x],
                   key=lambda x: x["mtime"])
    sessions, open_s = [], {}
    for a in items:
        key = (a["cam"], a["event"])
        s   = open_s.get(key)
        if s and a["mtime"] - s["last_mtime"] <= gap_min * 60:
            s["last_mtime"] = a["mtime"]
            s["last_time"]  = a["time"]
            s["last_url"]   = a["url"]
            s["photos"].append(a["url"])
        else:
            if s: sessions.append(s)
            open_s[key] = {
                "cam":        a["cam"],
                "event":      a["event"],
                "first_time": a["time"],
                "last_time":  a["time"],
                "first_url":  a["url"],
                "last_url":   a["url"],
                "first_mtime": a["mtime"],
                "last_mtime": a["mtime"],
                "photos":     [a["url"]],
            }
    for s in open_s.values(): sessions.append(s)

    for s in sessions:
        d = s["last_mtime"] - s["first_mtime"]
        m, sec = int(d // 60), int(d % 60)
        s["duration_sec"] = d
        s["duration_str"] = (f"{m} min {sec:02d} sec" if m > 0 else f"{sec} sec")
        s["duration_label"] = (f"{m}m {sec:02d}s" if m > 0 else f"{sec}s")

    return sorted(sessions, key=lambda x: x["last_mtime"], reverse=True)


# ── Routes ────────────────────────────────────────────────────────────────────
# NOTE: MJPEG removed — it holds a gunicorn thread permanently (1 per camera).
# Dashboard uses snapshot polling instead: each request ~15ms, thread released immediately.


# ── Snapshots ─────────────────────────────────────────────────────────────────
_last_snap = {}   # cam_name → jpeg bytes (used by annotation save)

def _snap_response(name, w, h, quality):
    stream = STREAMS.get(name)
    jpg = stream.jpeg(w, h, quality) if stream else None
    if jpg is None:
        f = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(f, name, (w//2-40, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50,50,50), 2)
        _, enc = cv2.imencode('.jpg', f); jpg = enc.tobytes()
    return jpg

@app.route("/snapshot/<name>")
def route_snapshot(name):
    """Standard snapshot (640×360) — used by dashboard fallback streams."""
    jpg = _snap_response(name, 640, 360, 72)
    _last_snap[name] = jpg
    resp = Response(jpg, mimetype="image/jpeg")
    resp.headers.update({"Cache-Control":"no-cache, no-store", "Pragma":"no-cache"})
    return resp

@app.route("/snapshot/<name>/hd")
def route_snapshot_hd(name):
    """HD snapshot (1280×720, quality 88) — used by annotation training page."""
    jpg = _snap_response(name, 1280, 720, 88)
    _last_snap[name] = jpg   # annotation saves this higher-res frame
    resp = Response(jpg, mimetype="image/jpeg")
    resp.headers.update({"Cache-Control":"no-cache, no-store", "Pragma":"no-cache"})
    return resp


# ── Annotation ────────────────────────────────────────────────────────────────

TRAIN_DIR  = Path("training_data")
CLASSES    = ["phone_hand","phone_ear","phone_desk","sleeping","working"]
CLASS_IDS  = {c:i for i,c in enumerate(CLASSES)}
_train_proc = None   # running train.py subprocess


def _annotation_counts():
    counts = {c: 0 for c in CLASSES}
    lbl_dir = TRAIN_DIR / "labels"
    if not lbl_dir.exists(): return counts
    for f in lbl_dir.glob("*.txt"):
        for line in f.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                idx = int(parts[0])
                if idx < len(CLASSES): counts[CLASSES[idx]] += 1
    return counts


@app.route("/api/annotate", methods=["POST"])
def api_save_annotation():
    from flask import request
    data = request.get_json()
    cam   = data.get("cam","")
    label = data.get("label","")
    box   = data.get("box",{})   # {x1,y1,x2,y2} normalized 0-1

    if label not in CLASSES:
        return jsonify({"error":"invalid label"}), 400

    # Use cached snapshot so saved image matches what user annotated
    jpg = _last_snap.get(cam)
    if not jpg:
        stream = STREAMS.get(cam)
        jpg = stream.jpeg(1280,720) if stream else None
    if not jpg:
        return jsonify({"error":"no frame"}), 400

    ts    = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19].replace(":","")
    fname = f"{cam}_{label}_{ts}"

    img_dir = TRAIN_DIR/"images"; img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir = TRAIN_DIR/"labels"; lbl_dir.mkdir(parents=True, exist_ok=True)

    # Save image
    img_path = img_dir / f"{fname}.jpg"
    with open(img_path,"wb") as f: f.write(jpg)

    # Save YOLO label  (class cx cy w h  — all normalized)
    lbl_path = lbl_dir / f"{fname}.txt"
    if label != "working":   # "working" = full negative, no box needed
        x1,y1,x2,y2 = box["x1"],box["y1"],box["x2"],box["y2"]
        cx = (x1+x2)/2;  cy = (y1+y2)/2
        bw = abs(x2-x1); bh = abs(y2-y1)
        with open(lbl_path,"w") as f:
            f.write(f"{CLASS_IDS[label]} {cx:.5f} {cy:.5f} {bw:.5f} {bh:.5f}\n")
    else:
        lbl_path.write_text("")  # empty = no objects visible

    counts = _annotation_counts()
    total  = sum(counts.values())
    _bust("counts")   # refresh annotation counts immediately
    print(f"  [Annotate] saved {fname}.jpg  (total: {total})")
    return jsonify({"success":True, "total":total, "counts":counts})


@app.route("/api/annotation_counts")
def api_annotation_counts():
    counts = _annotation_counts()
    total  = sum(counts.values())
    img_count = len(list((TRAIN_DIR/"images").glob("*.jpg"))) if (TRAIN_DIR/"images").exists() else 0
    return jsonify({"counts":counts,"total":total,"images":img_count})


@app.route("/api/train", methods=["POST"])
def api_start_train():
    global _train_proc
    import subprocess
    if _train_proc and _train_proc.poll() is None:
        return jsonify({"status":"running","message":"Training already in progress"})
    img_count = len(list((TRAIN_DIR/"images").glob("*.jpg"))) if (TRAIN_DIR/"images").exists() else 0
    if img_count < 15:
        return jsonify({"status":"error","message":f"Need 15+ images, have {img_count}"}),400
    _train_proc = subprocess.Popen(
        ["python3","train.py"],
        stdout=open("logs/train.log","w"), stderr=subprocess.STDOUT
    )
    return jsonify({"status":"started","images":img_count,"pid":_train_proc.pid})


@app.route("/api/train_status")
def api_train_status():
    global _train_proc
    custom = Path("custom_model/weights/best.pt")
    if _train_proc is None:
        return jsonify({"status":"idle","has_model":custom.exists()})
    rc = _train_proc.poll()
    if rc is None:
        # Read last few lines of log for progress
        log = Path("logs/train.log")
        progress = ""
        if log.exists():
            lines = log.read_text(errors="ignore").splitlines()
            progress = next((l for l in reversed(lines) if l.strip()), "")
        return jsonify({"status":"running","progress":progress,"has_model":custom.exists()})
    if rc == 0:
        return jsonify({"status":"done","has_model":custom.exists(),
                        "message":"Training complete! Restart monitor.py to use new model."})
    return jsonify({"status":"error","code":rc,"has_model":custom.exists()})


@app.route("/api/status")
def api_status():
    # Status is cheap — no caching needed
    return jsonify({c: {"connected": STREAMS[c].connected} for c in STREAMS})


@app.route("/api/alerts")
def api_alerts():
    def _compute():
        if not PHOTOS_DIR.exists(): return []
        photos = sorted(PHOTOS_DIR.glob("*.jpg"),
                        key=lambda x: x.stat().st_mtime, reverse=True)
        return _sessions(photos[:120])
    return jsonify(_cached("alerts", 4.0, _compute))


@app.route("/api/camera/<name>/alerts")
def api_camera_alerts(name):
    cam = next((c for c in CAMS if c["name"].upper() == name.upper()), None)
    if not cam: return jsonify([])
    prefix = f"Cam{cam['id']}_"
    def _compute():
        photos = sorted(
            [p for p in PHOTOS_DIR.glob("*.jpg") if p.name.startswith(prefix)],
            key=lambda x: x.stat().st_mtime, reverse=True)
        return _sessions(photos[:60])
    return jsonify(_cached(f"cam_alerts_{name}", 4.0, _compute))


@app.route("/api/logs")
def api_logs():
    def _compute():
        if not LOGS_FILE.exists(): return []
        lines = LOGS_FILE.read_text(errors="ignore").splitlines()
        clean = [l for l in lines if l.strip() and "[" in l]
        result = []
        for l in clean[-60:][::-1]:
            photo = None
            if "photo:" in l:
                try:
                    fname = Path(l.split("photo:")[-1].strip()).name
                    if fname and (PHOTOS_DIR / fname).exists():
                        photo = f"/photos/{fname}"
                except Exception:
                    pass
            result.append({"text": l, "photo": photo})
        return result
    return jsonify(_cached("logs", 5.0, _compute))


@app.route("/photos/<filename>")
def serve_photo(filename):
    return send_from_directory(str(PHOTOS_DIR.resolve()), filename)


# ── Dashboard HTML ────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Employee Monitor</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:#0b0b0b;color:#ddd;font-family:'Segoe UI',system-ui,sans-serif;
  height:100vh;display:flex;flex-direction:column;overflow:hidden}

/* Header */
.hdr{background:#121212;border-bottom:1px solid #222;padding:0 18px;
  height:50px;display:flex;align-items:center;justify-content:space-between;flex-shrink:0}
.hdr-l{display:flex;align-items:center;gap:12px}
.dot-live{width:9px;height:9px;border-radius:50%;background:#e74c3c;
  box-shadow:0 0 8px #e74c3c;animation:blink 1.8s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.hdr-title{font-size:16px;font-weight:700;color:#f0f0f0;letter-spacing:.4px}
.hdr-r{display:flex;align-items:center;gap:14px}
.pill{background:#1c1c1c;border:1px solid #282828;border-radius:20px;
  padding:3px 12px;font-size:11px;color:#888}
.pill b{margin-left:3px}
.pill.g b{color:#2ecc71}.pill.r b{color:#e74c3c}
#clock{font-size:14px;font-weight:600;color:#aaa;font-variant-numeric:tabular-nums}

/* Layout */
.body{display:flex;flex:1;overflow:hidden}

/* Camera panel */
.cam-panel{flex:1;overflow-y:auto;padding:12px;background:#0b0b0b}
.cam-panel::-webkit-scrollbar{width:4px}
.cam-panel::-webkit-scrollbar-thumb{background:#2a2a2a;border-radius:4px}
.cam-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}

.cam-card{background:#141414;border:1px solid #1e1e1e;border-radius:7px;
  overflow:hidden;cursor:pointer;transition:border-color .2s,transform .15s}
.cam-card:hover{border-color:#3a3a3a;transform:scale(1.01)}
.cam-card.alert{border-color:#c0392b;box-shadow:0 0 10px rgba(192,57,43,.35)}

.cam-hdr{display:flex;align-items:center;justify-content:space-between;
  padding:5px 9px;background:#0f0f0f;height:30px}
.cam-name{font-size:13px;font-weight:700;color:#e0e0e0;letter-spacing:.5px}
.cam-led{width:7px;height:7px;border-radius:50%;background:#333;transition:background .4s}
.cam-led.on{background:#2ecc71;box-shadow:0 0 5px #2ecc71}
.cam-led.off{background:#e74c3c}

.cam-feed{width:100%;display:block;aspect-ratio:16/9;
  background:#080808;object-fit:cover}

.cam-footer{padding:4px 9px;background:#0f0f0f;font-size:10.5px;
  color:#444;height:24px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cam-footer.ph{color:#e74c3c;font-weight:600}
.cam-footer.sl{color:#9b59b6;font-weight:600}

/* Side panel */
.side{width:310px;flex-shrink:0;background:#0f0f0f;
  border-left:1px solid #1e1e1e;display:flex;flex-direction:column;overflow:hidden}
.sec{display:flex;flex-direction:column;overflow:hidden}
.sec.alerts{flex:0 0 58%;border-bottom:1px solid #1e1e1e}
.sec.logs{flex:1}
.sec-title{padding:7px 13px;font-size:10px;font-weight:800;letter-spacing:1px;
  text-transform:uppercase;color:#444;background:#0b0b0b;border-bottom:1px solid #181818;
  flex-shrink:0}
.scroll{overflow-y:auto;flex:1}
.scroll::-webkit-scrollbar{width:3px}
.scroll::-webkit-scrollbar-thumb{background:#1e1e1e}

/* Alert cards */
.alert-card{display:flex;gap:9px;padding:8px 10px;border-bottom:1px solid #161616;
  cursor:pointer;transition:background .12s;align-items:flex-start}
.alert-card:hover{background:#161616}
.alert-thumb{width:76px;height:48px;object-fit:cover;border-radius:4px;
  border:1px solid #222;flex-shrink:0}
.alert-body{overflow:hidden;flex:1}
.alert-ev{font-size:11.5px;font-weight:700;color:#e74c3c;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.alert-ev.SL{color:#9b59b6}
.alert-dur{font-size:11px;color:#2ecc71;margin-top:2px;font-weight:600}
.alert-meta{font-size:10px;color:#444;margin-top:2px}
.alert-cam-tag{color:#4a90d9;font-weight:600}

/* Log rows */
.log-row{display:flex;align-items:center;gap:7px;padding:5px 10px;
  border-bottom:1px solid #141414;cursor:default}
.log-row:hover{background:#141414}
.log-thumb{width:44px;height:28px;object-fit:cover;border-radius:3px;
  border:1px solid #1e1e1e;flex-shrink:0;cursor:pointer}
.log-text{font-size:9.5px;font-family:'Consolas',monospace;color:#444;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;line-height:1.5}
.log-text .ep{color:#e74c3c}.log-text .es{color:#9b59b6}

/* Camera modal */
.modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.9);
  z-index:500;align-items:center;justify-content:center}
.modal.open{display:flex}
.modal-box{background:#141414;border:1px solid #2a2a2a;border-radius:10px;
  width:88vw;max-width:1100px;max-height:92vh;overflow:hidden;
  display:flex;flex-direction:column}
.modal-top{display:flex;align-items:center;justify-content:space-between;
  padding:10px 16px;background:#111;border-bottom:1px solid #222;flex-shrink:0}
.modal-cam-name{font-size:17px;font-weight:700;letter-spacing:.5px}
.modal-led{width:9px;height:9px;border-radius:50%;background:#2ecc71;
  box-shadow:0 0 6px #2ecc71;margin-right:6px;display:inline-block}
.close-btn{background:none;border:none;color:#666;font-size:22px;
  cursor:pointer;padding:2px 6px;border-radius:4px}
.close-btn:hover{color:#fff;background:#2a2a2a}
.modal-body{display:flex;flex:1;overflow:hidden}
.modal-stream-wrap{flex:1;background:#000;overflow:hidden}
.modal-stream-wrap img{width:100%;height:100%;object-fit:contain;display:block}
.modal-alerts-panel{width:260px;flex-shrink:0;overflow-y:auto;
  border-left:1px solid #1e1e1e;background:#0f0f0f}
.modal-alerts-panel::-webkit-scrollbar{width:3px}
.modal-alerts-panel::-webkit-scrollbar-thumb{background:#2a2a2a}
.m-sec-title{padding:8px 12px;font-size:10px;font-weight:800;letter-spacing:1px;
  text-transform:uppercase;color:#444;background:#0b0b0b;
  border-bottom:1px solid #181818;position:sticky;top:0;z-index:1}
.m-alert{padding:9px 12px;border-bottom:1px solid #161616;cursor:pointer}
.m-alert:hover{background:#161616}
.m-alert img{width:100%;border-radius:4px;margin-bottom:5px;
  border:1px solid #222;display:block}
.m-ev{font-size:11px;font-weight:700;color:#e74c3c}
.m-ev.SL{color:#9b59b6}
.m-dur{font-size:11px;color:#2ecc71;font-weight:600;margin-top:2px}
.m-times{font-size:10px;color:#444;margin-top:2px}

/* Photo lightbox */
#lb{display:none;position:fixed;inset:0;background:rgba(0,0,0,.95);
  z-index:1000;align-items:center;justify-content:center;flex-direction:column;gap:12px}
#lb.open{display:flex}
#lb img{max-width:90vw;max-height:80vh;border-radius:6px;border:1px solid #333}
#lb-info{color:#aaa;font-size:13px;text-align:center;line-height:1.7}
#lb-dur{color:#2ecc71;font-size:14px;font-weight:700}
#lb-close{position:absolute;top:16px;right:20px;font-size:26px;color:#666;
  cursor:pointer;background:none;border:none;padding:4px 8px}
#lb-close:hover{color:#fff}
</style>
</head>
<body>

<div class="hdr">
  <div class="hdr-l">
    <div class="dot-live"></div>
    <div class="hdr-title">Employee Monitor</div>
  </div>
  <div class="hdr-r">
    <div class="pill g">Live <b id="live-n">0</b></div>
    <div class="pill r">Alerts <b id="alert-n">0</b></div>
    <a href="/annotate" style="background:#1a3a1a;border:1px solid #2a5a2a;color:#2ecc71;padding:4px 14px;border-radius:20px;font-size:12px;text-decoration:none;font-weight:600">📚 Training</a>
    <div id="clock">--:--:--</div>
  </div>
</div>

<div class="body">

  <div class="cam-panel">
    <div class="cam-grid" id="cam-grid"></div>
  </div>

  <div class="side">
    <div class="sec alerts">
      <div class="sec-title">Recent Alerts &amp; Duration</div>
      <div class="scroll" id="alert-list">
        <div style="padding:18px;color:#333;font-size:11px;text-align:center">Loading...</div>
      </div>
    </div>
    <div class="sec logs">
      <div class="sec-title">Activity Log</div>
      <div class="scroll" id="log-list">
        <div style="padding:12px;color:#333;font-size:11px;text-align:center">Loading...</div>
      </div>
    </div>
  </div>
</div>

<!-- Camera modal -->
<div class="modal" id="cam-modal">
  <div class="modal-box">
    <div class="modal-top">
      <div style="display:flex;align-items:center">
        <span class="modal-led" id="modal-led"></span>
        <span class="modal-cam-name" id="modal-title">Camera</span>
      </div>
      <button class="close-btn" onclick="closeModal()">✕</button>
    </div>
    <div class="modal-body">
      <div class="modal-stream-wrap">
        <img id="modal-stream" src="" alt="stream">
      </div>
      <div class="modal-alerts-panel">
        <div class="m-sec-title">Alerts</div>
        <div id="modal-alert-list"></div>
      </div>
    </div>
  </div>
</div>

<!-- Lightbox -->
<div id="lb">
  <button id="lb-close" onclick="closeLb()">✕</button>
  <img id="lb-img" src="" alt="">
  <div id="lb-info"></div>
</div>

<script>
const CAMS   = CAMERA_LIST;
const CAMIDS = CAM_IDS;

/* ── Clock ─────────────────────────────────────────── */
setInterval(() => {
  document.getElementById('clock').textContent = new Date().toTimeString().slice(0,8)
}, 1000)

/* ── Build camera grid ─────────────────────────────── */
const grid = document.getElementById('cam-grid')
CAMS.forEach(name => {
  const div = document.createElement('div')
  div.className = 'cam-card'
  div.id = 'card-' + name
  div.innerHTML = `
    <div class="cam-hdr">
      <span class="cam-name">${name}</span>
      <span class="cam-led" id="led-${name}"></span>
    </div>
    <img class="cam-feed" id="feed-${name}" alt="${name}"
         style="background:#0a0a0a;cursor:pointer">
    <div class="cam-footer" id="foot-${name}">Connecting...</div>`
  div.querySelector('.cam-feed').onclick = () => openCam(name)
  div.querySelector('.cam-hdr').onclick  = () => openCam(name)
  grid.appendChild(div)
})

/* ── Snapshot polling — no MJPEG, no held threads ──────
   Each camera polls /snapshot/<name> every 600ms.
   Requests are staggered by 85ms so 7 cameras never
   hit the server simultaneously.
   Each request: ~15ms, thread released immediately.
   Never needs a browser refresh.
──────────────────────────────────────────────────────── */
const POLL_MS   = 600    // poll interval per camera
const STAGGER   = 85     // ms between each camera's poll

function fetchSnap(name) {
  const img = document.getElementById('feed-' + name)
  if (!img) return
  const t = new Image()
  t.onload  = () => { img.src = t.src }
  t.onerror = () => {}
  t.src = '/snapshot/' + name + '?_=' + Date.now()
}

function startPolling() {
  // First load: immediate, staggered
  CAMS.forEach((name, i) => setTimeout(() => fetchSnap(name), i * STAGGER))

  // Continuous polling loop, staggered
  setInterval(() => {
    CAMS.forEach((name, i) => setTimeout(() => fetchSnap(name), i * STAGGER))
  }, POLL_MS)
}
startPolling()

/* ── Camera status (every 3s) ──────────────────────── */
function pollStatus() {
  fetch('/api/status').then(r => r.json()).then(d => {
    let live = 0
    Object.entries(d).forEach(([n, v]) => {
      const led = document.getElementById('led-' + n)
      if (!led) return
      led.className = 'cam-led ' + (v.connected ? 'on' : 'off')
      if (v.connected) live++
    })
    document.getElementById('live-n').textContent = live
  }).catch(() => {})
}
setInterval(pollStatus, 3000); pollStatus()

/* ── Alerts (every 4s, immediate on load) ──────────── */
function evClass(ev) { return (ev.includes('SLEEP') || ev.includes('Slee')) ? 'SL' : '' }

function renderAlerts(sessions) {
  const el = document.getElementById('alert-list')
  document.getElementById('alert-n').textContent = sessions.length
  if (!sessions.length) {
    el.innerHTML = '<div style="padding:18px;color:#333;font-size:11px;text-align:center">No alerts yet</div>'
    return
  }
  el.innerHTML = sessions.map(s => `
    <div class="alert-card" onclick="openLb('${s.first_url}','${s.event}','${s.first_time}','${s.last_time}','${s.duration_str}')">
      <img class="alert-thumb" src="${s.last_url}?_=${s.last_mtime|0}" onerror="this.style.opacity=.3" alt="">
      <div class="alert-body">
        <div class="alert-ev ${evClass(s.event)}">${s.event}</div>
        <div class="alert-dur">⏱ ${s.duration_str}</div>
        <div class="alert-meta"><span class="alert-cam-tag">${s.cam}</span>
          &nbsp;${s.first_time}${s.first_time !== s.last_time ? ' → ' + s.last_time : ''}</div>
      </div>
    </div>`).join('')

  // Update camera card footers
  const byCam = {}
  sessions.forEach(s => { if (!byCam[s.cam]) byCam[s.cam] = s })
  CAMS.forEach((name, i) => {
    const key  = 'Cam' + CAMIDS[i]
    const s    = byCam[key]
    const foot = document.getElementById('foot-' + name)
    const card = document.getElementById('card-' + name)
    if (!foot) return
    if (s) {
      const isP = s.event.toUpperCase().includes('PHONE')
      const isS = s.event.toUpperCase().includes('SLEEP')
      foot.textContent = `⚠ ${s.event}  ${s.last_time}  (${s.duration_label || s.duration_str})`
      foot.className   = 'cam-footer ' + (isP ? 'ph' : isS ? 'sl' : '')
      if (card) card.className = 'cam-card ' + (isP || isS ? 'alert' : '')
    } else {
      foot.textContent = '● Live'
      foot.className   = 'cam-footer'
      if (card) card.className = 'cam-card'
    }
  })
}

function fetchAlerts() {
  fetch('/api/alerts').then(r => r.json()).then(renderAlerts).catch(() => {})
}
setInterval(fetchAlerts, 4000); fetchAlerts()

/* ── Logs (every 6s) ───────────────────────────────── */
function renderLogs(rows) {
  const el = document.getElementById('log-list')
  if (!rows.length) {
    el.innerHTML = '<div style="padding:12px;color:#333;font-size:11px;text-align:center">No log entries</div>'
    return
  }
  el.innerHTML = rows.map(r => {
    const safe = r.photo ? r.photo.replace(/'/g,"\\'") : ''
    const ph   = r.photo
      ? `<img class="log-thumb" src="${r.photo}" onclick="event.stopPropagation();openLb('${safe}','','','','')" onerror="this.style.opacity=.2" alt="">`
      : `<div style="width:44px;flex-shrink:0"></div>`
    const isP = r.text.includes('PHONE'); const isS = r.text.includes('SLEEP')
    const cls  = isP ? 'ep' : isS ? 'es' : ''
    const txt  = r.text.replace(/\|.*$/, '').replace('Employee detected ', '').substring(0, 85)
    return `<div class="log-row">${ph}<div class="log-text"><span class="${cls}">${txt}</span></div></div>`
  }).join('')
}

function fetchLogs() {
  fetch('/api/logs').then(r => r.json()).then(renderLogs).catch(() => {})
}
setInterval(fetchLogs, 6000); fetchLogs()

/* ── Camera modal ──────────────────────────────────── */
let activeCam = null, modalSnapTimer = null

function openCam(name) {
  activeCam = name
  document.getElementById('modal-title').textContent = name
  document.getElementById('cam-modal').classList.add('open')

  const mimg = document.getElementById('modal-stream')
  clearInterval(modalSnapTimer)

  // HD snapshot polling for modal — 400ms = 2.5fps at 1280×720
  function updateModal() {
    const t = new Image()
    t.onload = () => { mimg.src = t.src }
    t.src = '/snapshot/' + name + '/hd?_=' + Date.now()
  }
  updateModal()
  modalSnapTimer = setInterval(updateModal, 400)

  const led = document.getElementById('led-' + name)
  document.getElementById('modal-led').style.background =
    (led && led.classList.contains('on')) ? '#2ecc71' : '#888'

  loadCamAlerts(name)
}

function loadCamAlerts(name) {
  fetch('/api/camera/' + name + '/alerts').then(r => r.json()).then(sessions => {
    const el = document.getElementById('modal-alert-list')
    if (!sessions.length) {
      el.innerHTML = `<div style="padding:16px;color:#333;font-size:11px;text-align:center">No alerts for ${name}</div>`
      return
    }
    el.innerHTML = sessions.map(s => `
      <div class="m-alert" onclick="openLb('${s.first_url}','${s.event}','${s.first_time}','${s.last_time}','${s.duration_str}')">
        <img src="${s.last_url}" onerror="this.style.opacity=.3" alt="">
        <div class="m-ev ${evClass(s.event)}">${s.event}</div>
        <div class="m-dur">⏱ ${s.duration_str}</div>
        <div class="m-times">${s.first_time}${s.first_time !== s.last_time ? ' → ' + s.last_time : ''}</div>
      </div>`).join('')
  }).catch(() => {})
}

function closeModal() {
  document.getElementById('cam-modal').classList.remove('open')
  document.getElementById('modal-stream').src = ''
  clearInterval(modalSnapTimer); modalSnapTimer = null
  activeCam = null
}

/* ── Lightbox ──────────────────────────────────────── */
function openLb(url, ev, t1, t2, dur) {
  document.getElementById('lb-img').src = url
  let html = ev ? `<span style="color:#bbb;font-size:14px">${ev}</span>` : ''
  if (t1) html += `&ensp;First: <b style="color:#eee">${t1}</b>`
  if (t2 && t2 !== t1) html += `&ensp;→&ensp;Last: <b style="color:#eee">${t2}</b>`
  if (dur) html += `<br><span style="color:#2ecc71;font-size:16px;font-weight:700">⏱ ${dur}</span>`
  document.getElementById('lb-info').innerHTML = html
  document.getElementById('lb').classList.add('open')
}
function closeLb() {
  document.getElementById('lb').classList.remove('open')
  document.getElementById('lb-img').src = ''
}
document.getElementById('lb').addEventListener('click', e => {
  if (e.target === document.getElementById('lb')) closeLb()
})
document.getElementById('cam-modal').addEventListener('click', e => {
  if (e.target === document.getElementById('cam-modal')) closeModal()
})
document.addEventListener('keydown', e => { if (e.key==='Escape') { closeLb(); closeModal() } })
</script>
</body>
</html>"""


@app.route("/")
def dashboard():
    cam_names = [c["name"] for c in CAMS]
    cam_ids   = [c["id"]   for c in CAMS]
    html = (HTML
            .replace("CAMERA_LIST", json.dumps(cam_names))
            .replace("CAM_IDS",     json.dumps(cam_ids)))
    return html


# ── Annotation UI ─────────────────────────────────────────────────────────────

ANNOTATE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Training Annotation — Employee Monitor</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0b0b0b;color:#ddd;font-family:'Segoe UI',system-ui,sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}
.hdr{background:#121212;border-bottom:1px solid #222;padding:0 18px;height:50px;display:flex;align-items:center;justify-content:space-between;flex-shrink:0}
.hdr-title{font-size:16px;font-weight:700;color:#f0f0f0}
.back-btn{background:#1e1e1e;border:1px solid #333;color:#aaa;padding:6px 14px;border-radius:6px;cursor:pointer;font-size:13px;text-decoration:none}
.back-btn:hover{background:#2a2a2a;color:#fff}
.main{display:flex;flex:1;overflow:hidden;gap:0}
/* Left controls */
.controls{width:280px;flex-shrink:0;background:#111;border-right:1px solid #1e1e1e;padding:14px;display:flex;flex-direction:column;gap:12px;overflow-y:auto}
.ctrl-section{background:#161616;border:1px solid #1e1e1e;border-radius:8px;padding:12px}
.ctrl-title{font-size:10px;font-weight:800;letter-spacing:1px;text-transform:uppercase;color:#555;margin-bottom:10px}
select{width:100%;background:#0e0e0e;border:1px solid #2a2a2a;color:#ddd;padding:8px 10px;border-radius:6px;font-size:13px}
select:focus{outline:none;border-color:#4a90d9}
.freeze-row{display:flex;gap:8px}
.btn{width:100%;padding:9px;border:none;border-radius:6px;cursor:pointer;font-size:13px;font-weight:600;transition:background .15s}
.btn-freeze{background:#2a2a2a;color:#aaa}.btn-freeze:hover{background:#3a3a3a}
.btn-freeze.frozen{background:#e67e22;color:#fff}
.btn-live{background:#1e3a1e;color:#2ecc71;flex:1}.btn-live:hover{background:#2a4a2a}
.label-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.lbl-btn{padding:8px 6px;border:2px solid #2a2a2a;border-radius:6px;background:#141414;color:#888;cursor:pointer;font-size:11px;font-weight:600;text-align:center;transition:all .15s}
.lbl-btn:hover{border-color:#555;color:#ccc}
.lbl-btn.active{border-color:#4a90d9;color:#4a90d9;background:#0d1a2a}
.lbl-btn.ph{border-color:#e74c3c!important;color:#e74c3c!important;background:#1a0d0d!important}
.lbl-btn.sl{border-color:#9b59b6!important;color:#9b59b6!important;background:#150d1a!important}
.lbl-btn.wk{border-color:#2ecc71!important;color:#2ecc71!important;background:#0d1a0d!important}
.btn-save{background:#1a4a7a;color:#4ab3f4;margin-top:4px}.btn-save:hover{background:#1e5a90}
.btn-save:disabled{background:#1a1a1a;color:#444;cursor:not-allowed}
.btn-clear{background:#1a1a1a;color:#666}.btn-clear:hover{background:#2a2a2a;color:#aaa}
.count-row{display:flex;justify-content:space-between;font-size:11px;padding:3px 0;border-bottom:1px solid #1a1a1a}
.count-row:last-child{border-bottom:none}
.count-lbl{color:#666}.count-val{color:#aaa;font-weight:700}
.count-val.good{color:#2ecc71}
.btn-train{width:100%;padding:11px;border:none;border-radius:7px;cursor:pointer;font-size:14px;font-weight:700;background:#1a4a1a;color:#2ecc71;transition:all .2s}
.btn-train:hover:not(:disabled){background:#1f5f1f}
.btn-train:disabled{background:#141414;color:#333;cursor:not-allowed}
.train-progress{margin-top:8px;font-size:11px;color:#666;line-height:1.5;word-break:break-all}
.train-progress.running{color:#f39c12}
.train-progress.done{color:#2ecc71}
.train-progress.error{color:#e74c3c}
/* Camera feed area */
.feed-area{flex:1;display:flex;flex-direction:column;background:#0a0a0a;overflow:hidden}
.feed-wrap{flex:1;position:relative;overflow:hidden;cursor:crosshair;user-select:none}
img#snap{display:block;width:100%;height:100%;object-fit:contain;background:#0a0a0a}
canvas#canvas{position:absolute;top:0;left:0;width:100%;height:100%;cursor:crosshair}
.instructions{padding:10px 16px;background:#111;border-top:1px solid #1e1e1e;font-size:11.5px;color:#555;flex-shrink:0;line-height:1.8}
.instructions b{color:#888}
.toast{position:fixed;bottom:24px;left:50%;transform:translateX(-50%);background:#1a3a1a;border:1px solid #2ecc71;color:#2ecc71;padding:10px 22px;border-radius:8px;font-size:13px;font-weight:600;opacity:0;transition:opacity .3s;pointer-events:none;z-index:999}
.toast.show{opacity:1}
.model-badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:10px;font-weight:700;margin-top:4px}
.model-badge.custom{background:#1a3a1a;color:#2ecc71;border:1px solid #2ecc71}
.model-badge.base{background:#1a1a2a;color:#4a90d9;border:1px solid #4a90d9}
</style>
</head>
<body>

<div class="hdr">
  <div style="display:flex;align-items:center;gap:12px">
    <div style="width:9px;height:9px;border-radius:50%;background:#f39c12;box-shadow:0 0 6px #f39c12"></div>
    <span class="hdr-title">Training Annotation</span>
    <span id="model-badge" class="model-badge base">Base Model</span>
  </div>
  <a class="back-btn" href="/">← Live Dashboard</a>
</div>

<div class="main">
  <!-- Controls -->
  <div class="controls">

    <!-- Camera select -->
    <div class="ctrl-section">
      <div class="ctrl-title">Camera</div>
      <select id="cam-sel">CAM_OPTIONS</select>
      <div style="display:flex;gap:6px;margin-top:8px">
        <button class="btn btn-freeze" id="freeze-btn" onclick="toggleFreeze()">❄ Freeze</button>
        <button class="btn btn-live" onclick="setLive()">▶ Live</button>
      </div>
    </div>

    <!-- Labels -->
    <div class="ctrl-section">
      <div class="ctrl-title">Label — click to select</div>
      <div class="label-grid">
        <button class="lbl-btn ph" data-label="phone_hand" onclick="selectLabel(this)">📱 Phone in Hand</button>
        <button class="lbl-btn ph" data-label="phone_ear"  onclick="selectLabel(this)">📞 Phone on Ear</button>
        <button class="lbl-btn ph" data-label="phone_desk" onclick="selectLabel(this)">🖥 Phone on Desk</button>
        <button class="lbl-btn sl" data-label="sleeping"   onclick="selectLabel(this)">😴 Sleeping</button>
        <button class="lbl-btn wk" data-label="working"    onclick="selectLabel(this)" style="grid-column:span 2">✅ Working (negative)</button>
      </div>
    </div>

    <!-- Save / clear -->
    <div class="ctrl-section">
      <div class="ctrl-title">Actions</div>
      <button class="btn btn-save" id="save-btn" onclick="saveAnnotation()" disabled>💾 Save Annotation</button>
      <button class="btn btn-clear" style="margin-top:8px" onclick="clearBox()">✕ Clear Box</button>
    </div>

    <!-- Stats -->
    <div class="ctrl-section">
      <div class="ctrl-title">Dataset Progress</div>
      <div id="counts-list"></div>
    </div>

    <!-- Train -->
    <div class="ctrl-section">
      <div class="ctrl-title">Training</div>
      <button class="btn-train" id="train-btn" onclick="startTraining()" disabled>
        🚀 Train Model
      </button>
      <div class="train-progress" id="train-status"></div>
    </div>

  </div>

  <!-- Camera feed -->
  <div class="feed-area">
    <div class="feed-wrap" id="feed-wrap">
      <img id="snap" src="" alt="camera">
      <canvas id="canvas"></canvas>
    </div>
    <div class="instructions">
      <b>Step 1:</b> Select camera &nbsp;|&nbsp;
      <b>Step 2:</b> Click ❄ Freeze when you see the behavior &nbsp;|&nbsp;
      <b>Step 3:</b> Draw box around person &nbsp;|&nbsp;
      <b>Step 4:</b> Select label &nbsp;|&nbsp;
      <b>Step 5:</b> Click 💾 Save &nbsp;|&nbsp;
      Repeat 20+ times per class → click 🚀 Train Model
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
const CAMS = CAMERA_NAMES_LIST;
let frozen      = false;
let snapTimer   = null;
let currentCam  = CAMS[0] || '';
let selectedLbl = null;
let box         = null;   // {x1,y1,x2,y2} normalized 0-1
let drawing     = false;
let startX, startY;

// ── Snapshot loading ────────────────────────────────
function loadSnap() {
  if (frozen) return;
  // Use HD snapshot for annotation — better quality for drawing boxes
  document.getElementById('snap').src = '/snapshot/' + currentCam + '/hd?t=' + Date.now();
}
function startSnap() {
  clearInterval(snapTimer);
  loadSnap();
  snapTimer = setInterval(loadSnap, 2500);
}
function setLive() {
  frozen = false;
  document.getElementById('freeze-btn').textContent = '❄ Freeze';
  document.getElementById('freeze-btn').classList.remove('frozen');
  startSnap();
}
function toggleFreeze() {
  frozen = !frozen;
  const btn = document.getElementById('freeze-btn');
  if (frozen) {
    btn.textContent = '🔴 FROZEN'; btn.classList.add('frozen');
    clearInterval(snapTimer);
  } else {
    setLive();
  }
}

// ── Camera select ───────────────────────────────────
document.getElementById('cam-sel').addEventListener('change', e => {
  currentCam = e.target.value;
  box = null; redraw();
  setLive();
});
currentCam = document.getElementById('cam-sel').value;
startSnap();

// ── Label select ────────────────────────────────────
function selectLabel(btn) {
  document.querySelectorAll('.lbl-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  selectedLbl = btn.dataset.label;
  updateSaveBtn();
}

// ── Canvas drawing ──────────────────────────────────
const canvas = document.getElementById('canvas');
const ctx    = canvas.getContext('2d');

function syncCanvas() {
  const img  = document.getElementById('snap');
  const wrap = document.getElementById('feed-wrap');
  canvas.width  = wrap.clientWidth;
  canvas.height = wrap.clientHeight;
}
new ResizeObserver(syncCanvas).observe(document.getElementById('feed-wrap'));
syncCanvas();

function normCoord(ex, ey) {
  const r = canvas.getBoundingClientRect();
  return { x: (ex - r.left) / r.width, y: (ey - r.top) / r.height };
}

canvas.addEventListener('mousedown', e => {
  if (e.button !== 0) return;
  drawing = true;
  const c = normCoord(e.clientX, e.clientY);
  startX = c.x; startY = c.y;
  box = null; redraw();
});
canvas.addEventListener('mousemove', e => {
  if (!drawing) return;
  const c = normCoord(e.clientX, e.clientY);
  box = { x1: Math.min(startX,c.x), y1: Math.min(startY,c.y),
          x2: Math.max(startX,c.x), y2: Math.max(startY,c.y) };
  redraw();
});
canvas.addEventListener('mouseup', e => {
  drawing = false;
  if (box && (box.x2-box.x1 < 0.01 || box.y2-box.y1 < 0.01)) {
    box = null;  // too small
  }
  redraw(); updateSaveBtn();
});

// Touch support
canvas.addEventListener('touchstart', e => {
  e.preventDefault();
  const t = e.touches[0];
  canvas.dispatchEvent(new MouseEvent('mousedown', {clientX:t.clientX,clientY:t.clientY}));
});
canvas.addEventListener('touchmove', e => {
  e.preventDefault();
  const t = e.touches[0];
  canvas.dispatchEvent(new MouseEvent('mousemove', {clientX:t.clientX,clientY:t.clientY}));
});
canvas.addEventListener('touchend', e => {
  canvas.dispatchEvent(new MouseEvent('mouseup', {}));
});

function redraw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!box) return;
  const x = box.x1 * canvas.width;
  const y = box.y1 * canvas.height;
  const w = (box.x2 - box.x1) * canvas.width;
  const h = (box.y2 - box.y1) * canvas.height;

  // Shadow
  ctx.shadowColor = 'rgba(0,0,0,0.8)'; ctx.shadowBlur = 6;

  // Box
  ctx.strokeStyle = selectedLbl === 'sleeping' ? '#9b59b6' :
                    selectedLbl === 'working'  ? '#2ecc71' : '#e74c3c';
  ctx.lineWidth = 2.5;
  ctx.setLineDash([]);
  ctx.strokeRect(x, y, w, h);

  // Label
  if (selectedLbl) {
    ctx.shadowBlur = 0;
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fillRect(x, y - 22, 130, 22);
    ctx.fillStyle = '#fff';
    ctx.font = '12px Segoe UI, sans-serif';
    ctx.fillText(selectedLbl, x + 5, y - 6);
  }

  // Corner handles
  ctx.shadowBlur = 0;
  ctx.fillStyle = '#fff';
  [[x,y],[x+w,y],[x,y+h],[x+w,y+h]].forEach(([cx,cy]) => {
    ctx.fillRect(cx-3, cy-3, 6, 6);
  });
}

function clearBox() { box = null; redraw(); updateSaveBtn(); }
function updateSaveBtn() {
  document.getElementById('save-btn').disabled = !(box || selectedLbl === 'working') || !selectedLbl;
}

// ── Save annotation ─────────────────────────────────
async function saveAnnotation() {
  if (!selectedLbl) return showToast('Select a label first!', false);
  if (!box && selectedLbl !== 'working') return showToast('Draw a box first!', false);
  if (!frozen) toggleFreeze();   // auto-freeze if not already

  const payload = {
    cam:   currentCam,
    label: selectedLbl,
    box:   box || {x1:0,y1:0,x2:1,y2:1}
  };

  try {
    const r = await fetch('/api/annotate', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    const d = await r.json();
    if (d.success) {
      showToast(`✓ Saved! Total: ${d.total}`, true);
      clearBox();
      updateCounts(d.counts, d.total);
      setLive();   // auto-unfreeze for next annotation
    } else {
      showToast('Error: ' + (d.error || 'unknown'), false);
    }
  } catch(e) { showToast('Network error', false); }
}

// ── Counts / stats ──────────────────────────────────
function updateCounts(counts, total) {
  const el = document.getElementById('counts-list');
  const labels = ['phone_hand','phone_ear','phone_desk','sleeping','working'];
  const names  = ['Phone in Hand','Phone on Ear','Phone on Desk','Sleeping','Working'];
  el.innerHTML = labels.map((k,i) => {
    const v = (counts||{})[k] || 0;
    return `<div class="count-row">
      <span class="count-lbl">${names[i]}</span>
      <span class="count-val ${v>=20?'good':''}">${v}</span>
    </div>`
  }).join('') + `<div class="count-row" style="margin-top:6px;border-top:1px solid #2a2a2a;padding-top:6px">
    <span class="count-lbl" style="font-weight:700">Total images</span>
    <span class="count-val good">${total||0}</span>
  </div>`;
  document.getElementById('train-btn').disabled = (total < 15);
}

function fetchCounts() {
  fetch('/api/annotation_counts').then(r=>r.json()).then(d => {
    updateCounts(d.counts, d.images);
  }).catch(()=>{});
}
fetchCounts();

// ── Training ────────────────────────────────────────
let trainPolling = null;

async function startTraining() {
  const btn = document.getElementById('train-btn');
  btn.disabled = true;
  btn.textContent = '⏳ Starting...';
  try {
    const r = await fetch('/api/train', {method:'POST'});
    const d = await r.json();
    if (d.status === 'started') {
      showToast('Training started! ~30 min on GPU', true);
      pollTrainStatus();
    } else {
      showToast(d.message || 'Cannot start training', false);
      btn.disabled = false;
      btn.textContent = '🚀 Train Model';
    }
  } catch(e) { showToast('Error starting training', false); btn.disabled=false; }
}

function pollTrainStatus() {
  clearInterval(trainPolling);
  trainPolling = setInterval(async () => {
    try {
      const r = await fetch('/api/train_status');
      const d = await r.json();
      const el = document.getElementById('train-status');
      const btn = document.getElementById('train-btn');
      if (d.status === 'running') {
        el.className = 'train-progress running';
        el.textContent = '⏳ Training in progress...\n' + (d.progress||'');
        btn.textContent = '⏳ Training...'; btn.disabled = true;
      } else if (d.status === 'done') {
        el.className = 'train-progress done';
        el.textContent = '✅ ' + d.message;
        btn.textContent = '🚀 Train Again'; btn.disabled = false;
        clearInterval(trainPolling);
        document.getElementById('model-badge').textContent = 'Custom Model';
        document.getElementById('model-badge').className = 'model-badge custom';
        showToast('Training complete! Restart monitor.py', true);
      } else if (d.status === 'error') {
        el.className = 'train-progress error';
        el.textContent = '✗ Training failed. Check logs/train.log';
        btn.textContent = '🚀 Try Again'; btn.disabled = false;
        clearInterval(trainPolling);
      } else {
        clearInterval(trainPolling);
      }
    } catch(e) {}
  }, 5000);
}

// Resume polling if training was running
fetch('/api/train_status').then(r=>r.json()).then(d => {
  if (d.status === 'running') pollTrainStatus();
  if (d.has_model) {
    document.getElementById('model-badge').textContent = 'Custom Model';
    document.getElementById('model-badge').className = 'model-badge custom';
  }
});

// ── Toast ────────────────────────────────────────────
function showToast(msg, ok=true) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.style.background = ok ? '#1a3a1a' : '#3a1a1a';
  t.style.color = ok ? '#2ecc71' : '#e74c3c';
  t.style.borderColor = ok ? '#2ecc71' : '#e74c3c';
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2500);
}
</script>
</body>
</html>"""


@app.route("/annotate")
def annotate_page():
    cam_names = [c["name"] for c in CAMS]
    opts = "\n".join(f'<option value="{n}">{n}</option>' for n in cam_names)
    html = (ANNOTATE_HTML
            .replace("CAM_OPTIONS", opts)
            .replace("CAMERA_NAMES_LIST", json.dumps(cam_names)))
    return html


if __name__ == "__main__":
    n = len(CAMS)
    print(f"\n  Employee Monitor — Web Dashboard")
    print(f"  Cameras : {n}  ({', '.join(c['name'] for c in CAMS)})")
    print(f"  URL     : http://localhost:5000")
    print(f"  Tip     : Run alongside monitor.py\n")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
