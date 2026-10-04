"""Native Windows viewer for live_feed.py: starts the capture side in WSL and shows the cameras here.

Layout follows the rig: left mono (CAM_B) | RGB (CAM_A, middle) | right mono (CAM_C). Decoding and display run
natively, skipping WSLg's remote-desktop rendering. A reader thread always drains the socket and keeps only the
newest JPEG per camera, so the display never falls behind the cameras. Once a second the view is also written to
live/feed.jpg (with live/feed_stats.json) so the feed can be inspected without the window.

Usage:
    python view_feed.py                       # every connected camera, 30 fps with the RGB camera
    python view_feed.py --cams BC             # stereo pair only (60 fps at 640x400)
    python view_feed.py --rgb-size 1280x960   # extra options are passed to live_feed.py
Press 'q' or Esc to quit.
"""
import argparse
import json
import os
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

HEADER = struct.Struct("<BIfI")  # must match live_feed.py
NAMES = ("CAM_A", "CAM_B", "CAM_C")
DISPLAY_ORDER = ("CAM_B", "CAM_A", "CAM_C")
WSL_SCRIPT = "/mnt/c/Users/Ritik/Desktop/oakffc3p/live_feed.py"

parser = argparse.ArgumentParser()
parser.add_argument("--height", type=int, default=360, help="Tile height in pixels")
parser.add_argument("--port", type=int, default=5600)
parser.add_argument("--no-launch", action="store_true", help="live_feed.py --serve is already running in WSL")
args, passthrough = parser.parse_known_args()  # everything else goes to live_feed.py

capture = None
if not args.no_launch:
    cmd = (f"cd ~/oakffc3p && exec .venv/bin/python -u {WSL_SCRIPT} --serve --port {args.port} "
           + " ".join(passthrough) + " 2>&1 | grep --line-buffered -v 'USB protocol'")
    capture = subprocess.Popen(["wsl.exe", "-d", "Ubuntu-22.04", "--", "bash", "-c", cmd])


def connect():
    deadline = time.monotonic() + 120  # device boot + usbipd re-attach can take a while
    while time.monotonic() < deadline:
        if capture is not None and capture.poll() is not None:
            raise SystemExit(f"Capture process exited with code {capture.returncode}")
        try:
            s = socket.create_connection(("127.0.0.1", args.port), timeout=2)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
            s.settimeout(None)
            return s
        except OSError:
            time.sleep(0.5)
    raise SystemExit("Could not reach live_feed.py in WSL")


def recv_exact(s, n):
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = s.recv_into(view[got:], n - got)
        if not k:
            raise ConnectionError("stream closed")
        got += k
    return buf


latest, lock, new_frame, stop = {}, threading.Lock(), threading.Event(), threading.Event()


def reader(s):
    try:
        while not stop.is_set():
            cam, seq, latency, n = HEADER.unpack(recv_exact(s, HEADER.size))
            jpeg = recv_exact(s, n)
            with lock:
                latest[cam] = (seq, latency, time.perf_counter(), jpeg)
            new_frame.set()
    except (ConnectionError, OSError):
        stop.set()
        new_frame.set()


def write_atomic(path, write):
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    write(str(tmp))
    os.replace(tmp, path)


sock = connect()
threading.Thread(target=reader, args=(sock,), daemon=True).start()
print("Connected. Press 'q' in the window to quit.")

tiles, stats = {}, {}
t0, last_peek = time.monotonic(), 0.0
peek_dir = Path(__file__).resolve().parent / "live"
win = "OAK-FFC-3P live (left | RGB | right)"
cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

try:
    while not stop.is_set():
        new_frame.wait(0.05)
        new_frame.clear()
        with lock:
            items = dict(latest)
            latest.clear()
        for cam, (seq, latency, received, jpeg) in items.items():
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            name = NAMES[cam]
            tiles[name] = img
            st = stats.setdefault(name, {"n": 0, "fps": 0.0, "lat": 0.0, "host": 0.0, "size": None})
            st["n"] += 1
            st["lat"] = latency
            st["host"] = (time.perf_counter() - received) * 1000.0
            st["size"] = f"{img.shape[1]}x{img.shape[0]}"

        now = time.monotonic()
        if now - t0 >= 1.0:
            for st in stats.values():
                st["fps"], st["n"] = st["n"] / (now - t0), 0
            t0 = now

        if items and tiles:
            row = []
            for name in DISPLAY_ORDER:
                if name not in tiles:
                    continue
                img = tiles[name]
                img = cv2.resize(img, (round(img.shape[1] * args.height / img.shape[0]), args.height),
                                 interpolation=cv2.INTER_AREA)
                st = stats[name]
                label = {"CAM_A": "RGB (middle)", "CAM_B": "LEFT mono", "CAM_C": "RIGHT mono"}[name]
                text = f"{label}  {st['fps']:.0f} fps  {st['lat']:.0f}+{st['host']:.0f} ms"
                for c, t in (((0, 0, 0), 4), ((0, 255, 0), 1)):
                    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, t)
                row.append(img)
            view = np.hstack(row)
            cv2.imshow(win, view)
            if now - last_peek >= 1.0:
                last_peek = now
                peek_dir.mkdir(exist_ok=True)
                write_atomic(peek_dir / "feed.jpg", lambda f: cv2.imwrite(f, view, [cv2.IMWRITE_JPEG_QUALITY, 90]))
                summary = {n: {k: (round(v, 1) if isinstance(v, float) else v) for k, v in s.items() if k != "n"}
                           for n, s in stats.items()}
                write_atomic(peek_dir / "feed_stats.json",
                             lambda f: Path(f).write_text(json.dumps({"time": time.strftime("%H:%M:%S"), **summary})))
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_AUTOSIZE) < 0:  # -1 once the window is closed (minimized is fine)
            break
finally:
    stop.set()
    sock.close()
    if capture is not None:
        try:
            capture.wait(timeout=10)  # live_feed.py exits on disconnect, releasing the device
        except subprocess.TimeoutExpired:
            capture.kill()
