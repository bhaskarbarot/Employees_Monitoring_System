#!/usr/bin/env python3
"""
preview_cameras.py — See all cameras in a grid before recording.

Use this FIRST to:
  1. Check which cameras are connected
  2. See each camera's angle (ceiling / wall / corner)
  3. Decide the recording plan

Usage:
    python3 preview_cameras.py

Controls:
    Q — quit
"""

import os, cv2, json, threading, numpy as np, urllib.parse, time
from pathlib import Path

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


def build_rtsp(cam):
    pwd  = urllib.parse.quote(str(cam.get("pass", "")), safe="")
    base = f"rtsp://{cam.get('user','admin')}:{pwd}@{cam['ip']}:{cam.get('port',554)}"
    if cam.get("rtsp_path"): return base + cam["rtsp_path"]
    ch = cam.get("channel", 1)
    if cam.get("type", "dahua").lower() == "dahua":
        return base + f"/cam/realmonitor?channel={ch}&subtype=0"
    return base + f"/Streaming/Channels/{ch}01"


class CamFeed(threading.Thread):
    def __init__(self, cam):
        super().__init__(daemon=True)
        self.name_  = cam["name"]
        self.url    = build_rtsp(cam)
        self.frame  = None
        self.status = "Connecting..."
        self._lock  = threading.Lock()
        self.start()

    def run(self):
        while True:
            try:
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened():
                    self.status = "FAILED"
                    time.sleep(3); continue
                self.status = "Live"
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        self.status = "Reconnecting..."
                        break
                    with self._lock:
                        self.frame = frame
                cap.release()
            except Exception as e:
                self.status = f"Error: {e}"
            time.sleep(2)

    def get_frame(self):
        with self._lock:
            return self.frame.copy() if self.frame is not None else None


def main():
    cfg_file = Path("cameras.json")
    if not cfg_file.exists():
        print("cameras.json not found."); return

    with open(cfg_file) as f:
        cams = json.load(f)

    n    = len(cams)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    CW, CH = 640, 380

    print(f"\n  Connecting to {n} cameras...")
    feeds = [CamFeed(c) for c in cams]
    time.sleep(3)

    print(f"  Showing all {n} cameras. Press Q to quit.\n")

    while True:
        grid = np.zeros((rows * CH, cols * CW, 3), dtype=np.uint8)

        for i, feed in enumerate(feeds):
            r, c  = divmod(i, cols)
            y1, y2 = r*CH, (r+1)*CH
            x1, x2 = c*CW, (c+1)*CW

            frame = feed.get_frame()
            if frame is not None:
                cell = cv2.resize(frame, (CW, CH))
                status_col = (0, 220, 0)
            else:
                cell = np.zeros((CH, CW, 3), dtype=np.uint8)
                cv2.putText(cell, feed.status, (10, CH//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80,80,80), 1)
                status_col = (60, 60, 200)

            # Camera name bar at top
            cv2.rectangle(cell, (0, 0), (CW, 40), (20, 20, 20), -1)
            cv2.putText(cell, feed.name_, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_col, 2, cv2.LINE_AA)
            cv2.putText(cell, feed.status, (CW-140, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_col, 1, cv2.LINE_AA)

            grid[y1:y2, x1:x2] = cell

        cv2.imshow(f"Camera Preview — {n} cameras  [Q=quit]", grid)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("\n  Done. Now note which cameras have similar angles.")
    print("  Use that info to plan your training video recording.\n")


if __name__ == "__main__":
    main()
