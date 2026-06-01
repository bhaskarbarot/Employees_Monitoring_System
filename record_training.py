#!/usr/bin/env python3
"""
record_training.py — Record labelled training clips from any live camera.

Usage:
    python3 record_training.py --camera D2 --scenario phone_hand
    python3 record_training.py --camera D1 --scenario sleeping
    python3 record_training.py --camera D6 --scenario working

Scenarios:
    phone_hand   — person holding phone in hand
    phone_ear    — person with phone to ear
    phone_desk   — phone lying on desk, nobody using it
    sleeping     — person head down / sleeping at desk
    working      — normal work, NO phone visible  (negative samples)

Controls:
    SPACE        — start / stop recording
    Q / ESC      — quit

Output:
    training_videos/
        D2/
            phone_hand/  phone_hand_01.mp4  phone_hand_02.mp4 ...
            sleeping/    sleeping_01.mp4 ...
        D1/
            ...
"""

import os, cv2, json, time, argparse, urllib.parse
from pathlib import Path
from datetime import datetime

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

VALID_SCENARIOS = ["phone_hand", "phone_ear", "phone_desk", "sleeping", "working"]

# Recommended recording duration per scenario (seconds)
REC_GUIDE = {
    "phone_hand": "2–3 min  | show phone in both hands, different positions",
    "phone_ear":  "1–2 min  | left ear, right ear, head turning naturally",
    "phone_desk": "1–2 min  | phone flat on desk, person working normally",
    "sleeping":   "2–3 min  | stay still, different head positions",
    "working":    "3–4 min  | typing, mouse, writing — NO phone visible",
}


def load_camera(name):
    with open("cameras.json") as f:
        cams = json.load(f)
    for c in cams:
        if c["name"].upper() == name.upper():
            return c
    names = [c["name"] for c in cams]
    raise ValueError(f"Camera '{name}' not found. Available: {names}")


def build_rtsp(cam):
    pwd  = urllib.parse.quote(str(cam.get("pass", "")), safe="")
    base = f"rtsp://{cam.get('user','admin')}:{pwd}@{cam['ip']}:{cam.get('port',554)}"
    if cam.get("rtsp_path"): return base + cam["rtsp_path"]
    ch = cam.get("channel", 1)
    if cam.get("type","dahua").lower() == "dahua":
        return base + f"/cam/realmonitor?channel={ch}&subtype=0"
    return base + f"/Streaming/Channels/{ch}01"


def next_clip_number(cam_name, scenario):
    folder = Path("training_videos") / cam_name / scenario
    folder.mkdir(parents=True, exist_ok=True)
    return len(list(folder.glob("*.mp4"))) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera",   required=True,
                    help="Camera name e.g. D1 D2 D6 D9 D19 D31 D44")
    ap.add_argument("--scenario", required=True, choices=VALID_SCENARIOS,
                    help="What to record")
    args = ap.parse_args()

    cam = load_camera(args.camera)
    url = build_rtsp(cam)

    print(f"\n{'═'*60}")
    print(f"  Camera   : {cam['name']}")
    print(f"  Scenario : {args.scenario}")
    print(f"  Guide    : {REC_GUIDE[args.scenario]}")
    print(f"{'═'*60}")
    print(f"  SPACE = start/stop recording    Q = quit\n")

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"  ERROR: Cannot open camera {cam['name']}")
        print(f"  Check cameras.json — IP / credentials / channel number")
        return

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Save at 1280×720 max to keep file sizes manageable
    scale  = min(1.0, 1280 / max(src_w, 1))
    out_w  = int(src_w * scale)
    out_h  = int(src_h * scale)
    out_fps = min(src_fps, 25.0)

    print(f"  Source  : {src_w}×{src_h} @ {src_fps:.0f}fps")
    print(f"  Saving  : {out_w}×{out_h} @ {out_fps:.0f}fps")
    print(f"  Ready — press SPACE to start recording\n")

    recording   = False
    writer      = None
    rec_start   = None
    saved_path  = None
    clips_done  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("  Reconnecting..."); time.sleep(1); continue

        save_frame = cv2.resize(frame, (out_w, out_h)) if scale < 1.0 else frame.copy()
        display    = save_frame.copy()

        h, w = display.shape[:2]

        if recording:
            writer.write(save_frame)
            elapsed = time.time() - rec_start
            m, s    = int(elapsed // 60), int(elapsed % 60)

            # Red REC indicator
            cv2.circle(display, (32, 32), 14, (0, 0, 220), -1)
            cv2.putText(display, f"REC  {m:02d}:{s:02d}", (55, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 220), 2, cv2.LINE_AA)
            cv2.putText(display, f"{cam['name']}  |  {args.scenario}", (55, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(display, "SPACE = stop   Q = quit",
                        (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,120,120), 1)
        else:
            cv2.putText(display, f"{cam['name']}  |  {args.scenario}  |  READY",
                        (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2, cv2.LINE_AA)
            if clips_done:
                cv2.putText(display, f"Clips recorded: {clips_done}",
                            (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,200), 1)
            cv2.putText(display, "SPACE = start recording   Q = quit",
                        (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,120,120), 1)

        cv2.imshow(f"Training Recorder — {cam['name']} — {args.scenario}", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):   # Q or ESC
            break

        elif key == ord(' '):
            if not recording:
                # ── START ──────────────────────────────────────────
                num        = next_clip_number(cam["name"], args.scenario)
                saved_path = (Path("training_videos") / cam["name"] /
                              args.scenario / f"{args.scenario}_{num:02d}.mp4")
                fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
                writer     = cv2.VideoWriter(str(saved_path), fourcc, out_fps, (out_w, out_h))
                recording  = True
                rec_start  = time.time()
                print(f"  ● Recording  →  {saved_path}")

            else:
                # ── STOP ───────────────────────────────────────────
                recording = False
                elapsed   = time.time() - rec_start
                writer.release(); writer = None
                clips_done += 1
                print(f"  ■ Saved  {saved_path}  ({elapsed:.0f}s)")

                # Check how many clips done for this scenario
                total = next_clip_number(cam["name"], args.scenario) - 1
                needed = 5 if args.scenario != "working" else 3
                left   = max(0, needed - total)
                if left > 0:
                    print(f"     {total}/{needed} clips done — {left} more needed")
                    print(f"     Press SPACE for next clip\n")
                else:
                    print(f"     ✓ {total} clips done for {args.scenario} on {cam['name']}")
                    print(f"     Press Q to finish or SPACE to record more\n")

    # Cleanup
    if recording and writer:
        writer.release()
        elapsed = time.time() - rec_start
        print(f"  ■ Saved (on quit): {saved_path}  ({elapsed:.0f}s)")

    cap.release()
    cv2.destroyAllWindows()

    # Summary
    out_dir = Path("training_videos") / cam["name"] / args.scenario
    total = len(list(out_dir.glob("*.mp4")))
    print(f"\n  {cam['name']} / {args.scenario}: {total} clips total")
    print(f"  Folder: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
