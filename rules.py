"""
rules.py — Stateless alert handler.
Called by BehaviorEngine when a per-track threshold is exceeded.
Saves annotated photo + writes log entry + optionally calls Ollama.
"""

import os, time, base64, threading, cv2
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

USE_OLLAMA   = os.getenv("USE_OLLAMA",   "false").lower() == "true"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava:latest")

LOGS_DIR   = "logs"
PHOTOS_DIR = os.path.join(LOGS_DIR, "photos")
LOGS_FILE  = os.path.join(LOGS_DIR, "logs.txt")

os.makedirs(PHOTOS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)


_BANNER = {
    "PHONE_HAND": (20,  90, 200),
    "PHONE_EAR":  (0,   60, 220),
    "SLEEPING":   (40,  40, 160),
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


def fire_alert(event: str, frame, det: dict, elapsed_sec: float,
               cam_id: int = 1, cam_name: str = "Cam"):
    """
    Save annotated photo + write log line.
    det must contain: persons, phones, phone_violator, sleep_violator, waste_violator
    """
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname = f"Cam{cam_id}_{event}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    photo = os.path.join(PHOTOS_DIR, fname)

    annotated = _draw(frame.copy(), event, det, elapsed_sec, cam_name)
    cv2.imwrite(photo, annotated)

    m, s  = int(elapsed_sec//60), int(elapsed_sec%60)
    msg   = _DEFAULT_MSG[event]
    line  = (f"[{ts}]  [Cam{cam_id}:{cam_name}]  [{event}]  "
             f"{m}m {s:02d}s  —  {msg}  |  photo: {photo}\n")

    with open(LOGS_FILE, "a") as f:
        f.write(line)
    print(f"\033[91m[ALERT]\033[0m {line.strip()}")

    if USE_OLLAMA:
        threading.Thread(
            target=_ollama_update, args=(photo, event, line),
            daemon=True,
        ).start()


# ── Photo annotation ──────────────────────────────────────────────────────────

def _draw(frame, event, det, elapsed_sec, cam_name="Cam"):
    """
    Frame is already annotated with colored boxes from annotate_cam().
    Only add the event banner at top so saved photo clearly shows what was detected.
    """
    h, w = frame.shape[:2]

    # Resize to 1280×720 for consistent saved photo size
    if w != 1280 or h != 720:
        frame = cv2.resize(frame, (1280, 720))
        h, w = 720, 1280

    # Top banner — coloured background + event label
    m, s  = int(elapsed_sec // 60), int(elapsed_sec % 60)
    txt   = f"  [{cam_name}]  {_LABEL.get(event, event)}   |   {m} min {s:02d} sec"
    bh    = 80
    col   = _BANNER.get(event, (0, 0, 180))
    cv2.rectangle(frame, (0, 0), (w, bh), col, -1)
    scale = 1.4
    (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
    if tw > w - 20: scale *= (w - 20) / tw
    cv2.putText(frame, txt, (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)

    # Timestamp bottom-right
    ts_ = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame, ts_, (w - 310, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    return frame


# ── Ollama (background) ───────────────────────────────────────────────────────

def _ollama_update(photo, event, old_line):
    desc = _ollama_describe(photo, event)
    if desc:
        try:
            with open(LOGS_FILE,"r") as f: c = f.read()
            with open(LOGS_FILE,"w") as f:
                f.write(c.replace(_DEFAULT_MSG[event], desc))
        except: pass

def _ollama_describe(image_path, event):
    try:
        import requests
        with open(image_path,"rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        r = requests.post("http://localhost:11434/api/generate",
                          json={"model":OLLAMA_MODEL,
                                "prompt":_OLLAMA_PROMPT.get(event,"Describe what the employee is doing."),
                                "images":[b64],"stream":False},
                          timeout=20)
        if r.ok: return r.json().get("response","").strip()
    except: pass
    return ""
