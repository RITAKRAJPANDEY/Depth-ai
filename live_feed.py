"""Low-latency live feed from the OAK-FFC-3P cameras: stereo pair (CAM_B left, CAM_C right) + RGB (CAM_A, middle).

Frames are MJPEG-encoded on the device's hardware encoder, so ~1/20th of the raw data crosses USB, and every
frame is independent (no inter-frame buffering in the encoder or decoder). Host-side queues hold only the newest
frame, so a slow consumer drops frames instead of building up delay.

Runs inside WSL (Smart App Control blocks depthai on Windows). Two ways to view:
    python live_feed.py --serve        # stream JPEGs to view_feed.py running natively on Windows (lowest latency)
    python live_feed.py                # decode and show here, through WSLg
Options: --cams ABC (default: every connected camera), --res 800 (mono 1280x800), --rgb-size 1280x960, --fps N,
--quality 80. Press 'q' or Esc to quit.
"""
import argparse
import os
import socket
import struct
import threading
import time

os.environ.setdefault("DEPTHAI_BOOTUP_TIMEOUT", "60000")  # usbipd re-attaches the booted device after ~10 s
os.environ.setdefault("DEPTHAI_CONNECT_TIMEOUT", "60000")

import cv2
import depthai as dai
import numpy as np

HEADER = struct.Struct("<BIfI")  # camera index into NAMES, sequence number, sensor->host latency (ms), JPEG length
NAMES = ("CAM_A", "CAM_B", "CAM_C")
SOCKETS = (dai.CameraBoardSocket.CAM_A, dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C)
DISPLAY_ORDER = ("CAM_B", "CAM_A", "CAM_C")  # physical layout: left mono | middle RGB | right mono

parser = argparse.ArgumentParser()
parser.add_argument("--cams", default=None, help="Cameras to stream, e.g. ABC, BC, A (default: all connected)")
parser.add_argument("--res", type=int, default=400, choices=[400, 800], help="Mono: 400 = 640x400, 800 = 1280x800")
parser.add_argument("--rgb-size", default="960x720", help="RGB output size; 4:3 keeps the IMX577's full field of view")
parser.add_argument("--fps", type=float, default=None,
                    help="Default: 30 with the RGB camera, else 60 at 640x400 (20 for two cameras at 1280x800)")
parser.add_argument("--quality", type=int, default=80, help="MJPEG quality 1-100")
parser.add_argument("--max-exposure-us", type=int, default=8000,
                    help="Auto-exposure limit; short exposures avoid motion blur on a moving head (0 = no limit)")
parser.add_argument("--serve", action="store_true", help="Send JPEGs over TCP to view_feed.py instead of showing them")
parser.add_argument("--port", type=int, default=5600)
parser.add_argument("--height", type=int, default=360, help="Local view: tile height in pixels")
args = parser.parse_args()
MONO_SIZE = (1280, 800) if args.res == 800 else (640, 400)
RGB_SIZE = tuple(int(v) for v in args.rgb_size.lower().split("x"))


class LatestSlots:
    """Per-camera mailbox that keeps only the newest item."""

    def __init__(self):
        self.cv = threading.Condition()
        self.items = {}

    def put(self, key, item):
        with self.cv:
            self.items[key] = item
            self.cv.notify()

    def take(self, timeout=0.5):
        with self.cv:
            if not self.items:
                self.cv.wait(timeout)
            items, self.items = self.items, {}
            return items


def serve(slots, stop):
    server = socket.create_server(("0.0.0.0", args.port))
    print(f"Waiting for viewer on port {args.port}...", flush=True)
    conn, addr = server.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)  # a couple of frames at most in flight
    print(f"Viewer connected from {addr[0]}", flush=True)
    slots.take(0)  # drop anything captured while waiting
    try:
        while not stop.is_set():
            for cam, (seq, latency, jpeg) in sorted(slots.take().items()):
                conn.sendall(HEADER.pack(cam, seq, latency, len(jpeg)) + jpeg)
    except OSError:
        print("Viewer disconnected.", flush=True)
    finally:
        stop.set()
        conn.close()
        server.close()


def compose(tiles, stats, height):
    """Tiles side by side in physical order at a common height, with a status line on each."""
    row = []
    for name in DISPLAY_ORDER:
        if name not in tiles:
            continue
        img = tiles[name]
        img = cv2.resize(img, (round(img.shape[1] * height / img.shape[0]), height), interpolation=cv2.INTER_AREA)
        st = stats[name]
        text = f"{name} {'RGB' if name == 'CAM_A' else 'mono'}  {st['fps']:.0f} fps  {st['lat']:.0f} ms"
        for c, t in (((0, 0, 0), 4), ((0, 255, 0), 1)):
            cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, t)
        row.append(img)
    return np.hstack(row)


def show_local(slots, stop, active):
    tiles = {n: np.zeros((args.height, args.height * 4 // 3, 3), np.uint8) for n in active}
    stats = {n: {"n": 0, "fps": 0.0, "lat": 0.0} for n in active}
    t0 = time.monotonic()
    win = "OAK-FFC-3P live"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    while not stop.is_set():
        for cam, (seq, latency, jpeg) in slots.take(0.005).items():
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                tiles[NAMES[cam]] = img
                stats[NAMES[cam]]["n"] += 1
                stats[NAMES[cam]]["lat"] = latency
        now = time.monotonic()
        if now - t0 >= 1.0:
            for s in stats.values():
                s["fps"], s["n"] = s["n"] / (now - t0), 0
            t0 = now
        cv2.imshow(win, compose(tiles, stats, args.height))
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            stop.set()


def open_device(attempts=3):
    # usbipd occasionally misses the first re-attach of the booted device; booting again fixes it
    for attempt in range(1, attempts + 1):
        try:
            return dai.Device()
        except RuntimeError as e:
            if attempt == attempts:
                raise
            print(f"Device not reachable after boot ({e}), retrying ({attempt}/{attempts - 1})...", flush=True)
            time.sleep(3)


device = open_device()
connected = {s.name for s in device.getConnectedCameras()}
wanted = args.cams.upper() if args.cams else "".join(n[-1] for n in NAMES if n in connected)
active = [n for n in NAMES if n[-1] in wanted and n in connected]
missing = [f"CAM_{c}" for c in wanted if f"CAM_{c}" not in connected]
if missing:
    print(f"WARNING: requested but not detected: {', '.join(missing)}", flush=True)
if args.fps is None:
    # the device converts ~47 full-res mono frames/s to NV12 for the encoder; above that frames queue on the device
    args.fps = 30 if "CAM_A" in active else (60 if args.res == 400 else (20 if len(active) > 1 else 30))

with dai.Pipeline(device) as pipeline:
    queues = []  # (index into NAMES, queue)
    for index, (name, socket_id) in enumerate(zip(NAMES, SOCKETS)):
        if name not in active:
            continue
        cam = pipeline.create(dai.node.Camera).build(socket_id)
        if args.max_exposure_us:
            cam.initialControl.setAutoExposureLimit(args.max_exposure_us)
        size = RGB_SIZE if name == "CAM_A" else MONO_SIZE
        out = cam.requestOutput(size, type=dai.ImgFrame.Type.NV12, fps=args.fps)
        enc = pipeline.create(dai.node.VideoEncoder).build(
            out, frameRate=args.fps, profile=dai.VideoEncoderProperties.Profile.MJPEG, quality=args.quality)
        queues.append((index, enc.out.createOutputQueue(maxSize=1, blocking=False)))

    pipeline.start()
    sensors = {f.socket.name: f.sensorName for f in device.getConnectedCameraFeatures()}
    print(f"USB {device.getUsbSpeed().name} | " + " | ".join(
        f"{n} {sensors.get(n, '?')} {RGB_SIZE if n == 'CAM_A' else MONO_SIZE}" for n in active) +
        f" | {args.fps:g} fps, MJPEG q{args.quality}", flush=True)

    slots, stop = LatestSlots(), threading.Event()

    def pump():
        while pipeline.isRunning() and not stop.is_set():
            got = False
            for i, q in queues:
                msgs = q.tryGetAll()
                if msgs:
                    m = msgs[-1]
                    latency = (dai.Clock.now() - m.getTimestamp()).total_seconds() * 1000.0
                    slots.put(i, (m.getSequenceNum(), latency, m.getData().tobytes()))
                    got = True
            if not got:
                time.sleep(0.0005)
        stop.set()

    if args.serve:
        threading.Thread(target=serve, args=(slots, stop), daemon=True).start()
        pump()
    else:  # OpenCV windows must live on the main thread
        threading.Thread(target=pump, daemon=True).start()
        show_local(slots, stop, active)
