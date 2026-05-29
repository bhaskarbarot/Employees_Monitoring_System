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

_GREEN  = (0, 210,   0)
_RED    = (0,   0, 220)
_ORANGE = (0, 140, 255)

_BANNER = {
    "PHONE_HAND":   (20,  90, 200),
    "PHONE_EAR":    (0,   60, 220),
    "SLEEPING":     (40,  40, 160),
    "TIME_WASTING": (30, 130,  30),
}
_LABEL = {
    "PHONE_HAND":   "PHONE IN HAND",
    "PHONE_EAR":    "PHONE ON EAR / CALLING",
    "SLEEPING":     "SLEEPING / HEAD ON DESK",
    "TIME_WASTING": "TIME WASTING",
}
_DEFAULT_MSG = {
    "PHONE_HAND":   "Employee detected holding phone in hand.",
    "PHONE_EAR":    "Employee detected talking on phone / phone held to ear.",
    "SLEEPING":     "Employee appears sleeping — head down, no face visible, or motionless.",
    "TIME_WASTING": "Employee not working — standing away, looking sideways, or leaning back.",
}
_OLLAMA_PROMPT = {
    "PHONE_HAND":   "In one sentence, describe what the employee is doing with the phone in their hand.",
    "PHONE_EAR":    "In one sentence, describe the employee talking on or holding a phone to their ear.",
    "SLEEPING":     "In one sentence, describe the employee's posture suggesting they are sleeping.",
    "TIME_WASTING": "In one sentence, describe what the employee is doing instead of working.",
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
    h, w = frame.shape[:2]

    violator = (det.get("phone_violator") or
                det.get("sleep_violator") or
                det.get("waste_violator"))

    # Person boxes
    for pe in det.get("persons", []):
        x1,y1,x2,y2 = pe["bbox"]
        is_v  = (pe["bbox"] == violator)
        color = _RED if is_v else _GREEN
        label = _LABEL.get(event,"VIOLATION") if is_v else "Working"
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,3 if is_v else 2)
        (lw,lh),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
        ly = max(y1-4,lh+4)
        cv2.rectangle(frame,(x1,ly-lh-3),(x1+lw+4,ly+2),color,-1)
        cv2.putText(frame,label,(x1+2,ly-1),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2,cv2.LINE_AA)

    # Phone boxes
    for ph in det.get("phones",[]):
        x1,y1,x2,y2 = ph["bbox"]
        cv2.rectangle(frame,(x1,y1),(x2,y2),_ORANGE,2)
        cv2.putText(frame,"PHONE",(x1+2,max(y1-4,16)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,_ORANGE,2,cv2.LINE_AA)

    # Top banner
    m,s  = int(elapsed_sec//60), int(elapsed_sec%60)
    txt  = f"  [{cam_name}]  {_LABEL.get(event,event)}   |   {m} min {s:02d} sec"
    bh   = 90
    col  = _BANNER.get(event,(0,0,180))
    cv2.rectangle(frame,(0,0),(w,bh),col,-1)
    scale = 1.6
    (tw,_),_ = cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,scale,3)
    if tw > w-20: scale *= (w-20)/tw
    cv2.putText(frame,txt,(10,65),cv2.FONT_HERSHEY_SIMPLEX,scale,
                (255,255,255),3,cv2.LINE_AA)

    # Timestamp
    ts_ = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame,ts_,(w-310,h-12),cv2.FONT_HERSHEY_SIMPLEX,
                0.6,(200,200,200),1,cv2.LINE_AA)
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
