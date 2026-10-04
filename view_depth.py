"""Native Windows viewer for depth_live.py: starts the depth pipeline in WSL and shows it here.

Left: rectified CAM_B image. Right: colorized depth (near = red, far = blue, black = no depth).
The crosshair shows the median distance at the center; click anywhere to measure that point.
A reader thread always drains the socket and keeps only the newest frame, so the display never falls behind.

Usage:
    python view_depth.py                # 640x400 depth @ 30 fps (~70 ms camera->PC)
    python view_depth.py --near         # extended disparity, measures down to ~16 cm (more noise)
    python view_depth.py --max-m 3      # color scale range (colors follow log distance)
    python view_depth.py --conf 200     # extra options are passed to depth_live.py
Keys: q/Esc quit, s save a snapshot (PNG image + 16-bit depth) next to this script.
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

HEADER = struct.Struct("<BIfIHH")  # must match depth_live.py (kind 0 = left JPEG, 1 = depth, 2 = IMU JSON)
WSL_SCRIPT = "/mnt/c/Users/Ritik/Desktop/oakffc3p/depth_live.py"

parser = argparse.ArgumentParser()
parser.add_argument("--fps", type=float, default=30, help="Above ~34 fps the device can't keep up and latency jumps")
parser.add_argument("--near", action="store_true", help="Extended disparity: halves the minimum distance")
parser.add_argument("--temporal", action="store_true", help="Smoother depth on static scenes, lags on motion")
parser.add_argument("--min-m", type=float, default=0.25, help="Distance at the near (red) end of the color scale")
parser.add_argument("--max-m", type=float, default=5.0, help="Distance at the far (blue) end of the color scale")
parser.add_argument("--port", type=int, default=5601)
parser.add_argument("--peek-every", type=float, default=1.0,
                    help="Seconds between copies of the view written to live/ for remote inspection (0 = off)")
parser.add_argument("--no-launch", action="store_true", help="depth_live.py is already running in WSL")
args, passthrough = parser.parse_known_args()  # anything else goes to depth_live.py

capture = None
if not args.no_launch:
    flags = f"--port {args.port} --fps {args.fps:g}" + (" --near" if args.near else "") + \
            (" --temporal" if args.temporal else "") + "".join(f" {a}" for a in passthrough)
    cmd = f"cd ~/oakffc3p && exec .venv/bin/python -u {WSL_SCRIPT} {flags} 2>&1 | grep --line-buffered -v 'USB protocol'"
    capture = subprocess.Popen(["wsl.exe", "-d", "Ubuntu-22.04", "--", "bash", "-c", cmd])


def connect():
    deadline = time.monotonic() + 120  # device boot + usbipd re-attach can take a while
    while time.monotonic() < deadline:
        if capture is not None and capture.poll() is not None:
            raise SystemExit(f"Depth process exited with code {capture.returncode}")
        try:
            s = socket.create_connection(("127.0.0.1", args.port), timeout=2)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
            s.settimeout(None)
            return s
        except OSError:
            time.sleep(0.5)
    raise SystemExit("Could not reach depth_live.py in WSL")


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
            kind, seq, latency, n, w, h = HEADER.unpack(recv_exact(s, HEADER.size))
            payload = recv_exact(s, n)
            with lock:
                latest[kind] = (latency, w, h, payload)
            new_frame.set()
    except (ConnectionError, OSError):
        stop.set()
        new_frame.set()


# depth (mm) -> BGR lookup table: TURBO colormap on log distance, so every doubling of distance gets the same share
# of the colors (near = red, far = blue, 0 = black)
mm = np.maximum(np.arange(65536, dtype=np.float64), 1.0)
near, far = args.min_m * 1000.0, args.max_m * 1000.0
level = 1.0 - np.clip(np.log(mm / near) / np.log(far / near), 0, 1)
lut = cv2.applyColorMap((level * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
lut[0] = 0

click = {"pt": None}


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        click["pt"] = (x % 640, y)  # either half maps to the same pixel
    elif event == cv2.EVENT_RBUTTONDOWN:
        click["pt"] = None


def median_mm(depth, x, y, r=5):
    patch = depth[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
    valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size else None


def label(img, x, y, mm):
    text = f"{mm / 1000:.3f} m" if mm else "no depth"
    cv2.drawMarker(img, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
    for color, thick in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(img, text, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thick)


sock = connect()
threading.Thread(target=reader, args=(sock,), daemon=True).start()
print("Connected. Click to measure, right-click to clear, 's' to save, 'q' to quit.")

win = "OAK-FFC-3P depth (rectified left | depth)"
cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(win, on_mouse)
left = np.zeros((400, 640), np.uint8)
depth = np.zeros((400, 640), np.uint16)
imu = None
counts, shown_fps, lat, t0 = 0, 0.0, 0.0, time.monotonic()
snap_dir = Path(__file__).resolve().parent / "snapshots"
peek_dir = Path(__file__).resolve().parent / "live"
last_peek = 0.0


def write_atomic(path, write):
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    write(str(tmp))
    os.replace(tmp, path)


def peek(view, left, depth, stats):
    """Latest view, raw frames and stats in live/, so the feed can be inspected without the window."""
    peek_dir.mkdir(exist_ok=True)
    write_atomic(peek_dir / "view.jpg", lambda f: cv2.imwrite(f, view, [cv2.IMWRITE_JPEG_QUALITY, 90]))
    write_atomic(peek_dir / "left.png", lambda f: cv2.imwrite(f, left))
    write_atomic(peek_dir / "depth_mm.png", lambda f: cv2.imwrite(f, depth))
    write_atomic(peek_dir / "stats.json", lambda f: Path(f).write_text(json.dumps(stats, indent=1)))

try:
    while not stop.is_set():
        new_frame.wait(0.05)
        new_frame.clear()
        with lock:
            items = dict(latest)
            latest.clear()
        if 0 in items:
            img = cv2.imdecode(np.frombuffer(items[0][3], np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                left = img
        if 2 in items:
            imu = json.loads(bytes(items[2][3]))
        if 1 in items:
            lat, w, h, payload = items[1]
            depth = np.frombuffer(payload, np.uint16).reshape(h, w)
            counts += 1
        now = time.monotonic()
        if now - t0 >= 1.0:
            shown_fps, counts, t0 = counts / (now - t0), 0, now

        if items:
            color = lut[depth]
            if depth.shape != left.shape:  # decimated depth (e.g. while recording): show it at image size
                depth = cv2.resize(depth, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_NEAREST)
                color = lut[depth]
            left_bgr = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
            h, w = depth.shape
            for x, y in [(w // 2, h // 2)] + ([click["pt"]] if click["pt"] else []):
                mm = median_mm(depth, x, y)
                label(color, x, y, mm)
                label(left_bgr, x, y, mm)
            if imu:
                r, p, y = imu["roll_pitch_yaw_deg"]
                imu_text = (f"IMU roll {r:+.0f} pitch {p:+.0f} yaw {y:+.0f}  {imu['gyro_dps']:.0f} deg/s  "
                            f"blur {imu['blur_px']:.1f} px" + ("  MOTION BLUR" if imu["blurry"] else ""))
                imu_color = (0, 0, 255) if imu["blurry"] else (0, 255, 0)
                for c, t in (((0, 0, 0), 4), (imu_color, 1)):
                    cv2.putText(left_bgr, imu_text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, t)
                if imu["blurry"]:
                    cv2.rectangle(left_bgr, (1, 1), (left_bgr.shape[1] - 2, left_bgr.shape[0] - 2), (0, 0, 255), 3)
            text = f"{shown_fps:.0f} fps  cam->PC {lat:.0f} ms  fill {100.0 * (depth > 0).mean():.0f}%"
            for c, t in (((0, 0, 0), 4), ((0, 255, 0), 1)):
                cv2.putText(color, text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, t)
            view = np.hstack([left_bgr, color])
            cv2.imshow(win, view)
            if args.peek_every and now - last_peek >= args.peek_every:
                last_peek = now
                valid = depth[depth > 0]
                peek(view, left, depth, {
                    "time": time.strftime("%H:%M:%S"), "fps": round(shown_fps, 1), "latency_ms": round(lat, 1),
                    "fill_pct": round(100.0 * valid.size / depth.size, 1),
                    "left_brightness": round(float(left.mean()), 1),
                    "left_sharpness": round(float(cv2.Laplacian(left, cv2.CV_64F).var()), 1),
                    "depth_mm_p5_p50_p95": [int(v) for v in np.percentile(valid, [5, 50, 95])] if valid.size else None,
                    "center_mm": median_mm(depth, depth.shape[1] // 2, depth.shape[0] // 2),
                    "click": click["pt"],
                    "imu": imu,
                })

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_AUTOSIZE) < 0:  # -1 once the window is closed (minimized is fine)
            break
        if key == ord("s"):
            snap_dir.mkdir(exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(snap_dir / f"{stamp}_left.png"), left)
            cv2.imwrite(str(snap_dir / f"{stamp}_depth_mm.png"), depth)
            print(f"Saved snapshot {stamp} to {snap_dir}")
finally:
    stop.set()
    sock.close()
    if capture is not None:
        try:
            capture.wait(timeout=10)  # depth_live.py exits on disconnect, releasing the device
        except subprocess.TimeoutExpired:
            capture.kill()
