"""Native Windows RGB + depth viewer for the OAK-FFC-3P (IMX577 RGB on CAM_A, OV9282 stereo on CAM_B/CAM_C).

Left: RGB (CAM_A). Right: colorized stereo depth aligned to the RGB view (near = red, far = blue, black = no depth).
Depth is computed on the device from 640x400 mono frames and re-projected into the RGB camera with the calibration
in the device EEPROM; the RGB is MJPEG-encoded on the device. That keeps the stream small enough for USB2
(~25 MB/s usable), and every queue keeps only the newest frame so the view never falls behind.

Usage:
    python view_rgbd.py                 # 20 fps
    python view_rgbd.py --fps 25 --near # extended disparity: measures down to ~20 cm (more noise)
    python view_rgbd.py --max-m 3       # color scale range (colors follow log distance)
Keys: o toggle depth overlay on the RGB, s save a snapshot (RGB + 16-bit depth), q/Esc quit.
Click to measure a point (either half), right-click to clear.
"""
import argparse
import json
import os
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

SIZE = (640, 480)        # RGB / aligned depth output (4:3, full IMX577 field of view)
MONO_SIZE = (640, 400)   # stereo input (full OV9282 field of view, 2x2 binned)

parser = argparse.ArgumentParser()
parser.add_argument("--fps", type=float, default=20, help="USB2 keeps up to ~25 fps with RGB + depth")
parser.add_argument("--near", action="store_true", help="Extended disparity: halves the minimum distance")
parser.add_argument("--min-m", type=float, default=0.3, help="Distance at the near (red) end of the color scale")
parser.add_argument("--max-m", type=float, default=5.0, help="Distance at the far (blue) end of the color scale")
parser.add_argument("--max-exposure-us", type=int, default=8000,
                    help="Mono auto-exposure limit; short exposures avoid motion blur (0 = no limit)")
parser.add_argument("--quality", type=int, default=85, help="MJPEG quality of the RGB stream")
parser.add_argument("--peek-every", type=float, default=1.0,
                    help="Seconds between copies of the view written to live/ for remote inspection (0 = off)")
args = parser.parse_args()

# depth (mm) -> BGR lookup table: TURBO on log distance (near = red, far = blue, 0 = black), same as view_depth.py
mm = np.maximum(np.arange(65536, dtype=np.float64), 1.0)
near, far = args.min_m * 1000.0, args.max_m * 1000.0
level = 1.0 - np.clip(np.log(mm / near) / np.log(far / near), 0, 1)
lut = cv2.applyColorMap((level * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
lut[0] = 0

click = {"pt": None}


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        click["pt"] = (x % SIZE[0], y)  # either half maps to the same pixel
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


def text(img, s, y):
    for color, thick in (((0, 0, 0), 4), ((0, 255, 0), 1)):
        cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thick)


def write_atomic(path, write):
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    write(str(tmp))
    os.replace(tmp, path)


A, B, C = dai.CameraBoardSocket.CAM_A, dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C
device = dai.Device()
calib = device.readCalibration()
print(f"USB {device.getUsbSpeed().name}, baseline {calib.getBaselineDistance(C, B, useSpecTranslation=False):.3f} cm, "
      f"right->rgb {calib.getBaselineDistance(A, C, useSpecTranslation=False):.3f} cm", flush=True)

with dai.Pipeline(device) as pipeline:
    rgb = pipeline.create(dai.node.Camera).build(A)
    left = pipeline.create(dai.node.Camera).build(B)
    right = pipeline.create(dai.node.Camera).build(C)
    if args.max_exposure_us:
        for cam in (left, right):
            cam.initialControl.setAutoExposureLimit(args.max_exposure_us)

    stereo = pipeline.create(dai.node.StereoDepth).build(
        left.requestOutput(MONO_SIZE, fps=args.fps),
        right.requestOutput(MONO_SIZE, fps=args.fps),
        presetMode=dai.node.StereoDepth.PresetMode.DEFAULT,
    )
    stereo.setDepthAlign(A)              # re-project depth into the RGB camera
    stereo.setOutputSize(*SIZE)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setExtendedDisparity(args.near)
    stereo.setRectifyEdgeFillColor(0)
    pp = stereo.initialConfig.postProcessing
    pp.speckleFilter.enable = True
    pp.speckleFilter.speckleRange = 50
    pp.thresholdFilter.minRange = 100
    pp.thresholdFilter.maxRange = 15000

    rgb_out = rgb.requestOutput(SIZE, type=dai.ImgFrame.Type.NV12, fps=args.fps)
    enc = pipeline.create(dai.node.VideoEncoder).build(
        rgb_out, frameRate=args.fps, profile=dai.VideoEncoderProperties.Profile.MJPEG, quality=args.quality)

    q_depth = stereo.depth.createOutputQueue(maxSize=1, blocking=False)
    q_rgb = enc.out.createOutputQueue(maxSize=1, blocking=False)
    pipeline.start()
    print("Running. Click to measure, right-click to clear, 'o' overlay, 's' save, 'q' quit.", flush=True)

    win = "OAK-FFC-3P  RGB | depth (aligned to RGB)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    frame = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    depth = np.zeros((SIZE[1], SIZE[0]), np.uint16)
    overlay = False
    lat_rgb = lat_depth = 0.0
    counts, shown_fps, t0, last_peek = 0, 0.0, time.monotonic(), 0.0
    snap_dir = Path(__file__).resolve().parent / "snapshots"
    peek_dir = Path(__file__).resolve().parent / "live"

    while pipeline.isRunning():
        m_rgb, m_depth = q_rgb.tryGet(), q_depth.tryGet()
        if m_rgb is not None:
            img = cv2.imdecode(np.frombuffer(m_rgb.getData(), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                frame = img
                lat_rgb = (dai.Clock.now() - m_rgb.getTimestamp()).total_seconds() * 1000
        if m_depth is not None:
            depth = m_depth.getFrame()
            if depth.shape[::-1] != SIZE:
                depth = cv2.resize(depth, SIZE, interpolation=cv2.INTER_NEAREST)
            lat_depth = (dai.Clock.now() - m_depth.getTimestamp()).total_seconds() * 1000
            counts += 1
        now = time.monotonic()
        if now - t0 >= 1.0:
            shown_fps, counts, t0 = counts / (now - t0), 0, now

        if m_rgb is not None or m_depth is not None:
            color = lut[depth]
            view_rgb = frame.copy()
            if overlay:
                mask = depth > 0
                view_rgb[mask] = cv2.addWeighted(view_rgb, 0.45, color, 0.55, 0)[mask]
            h, w = depth.shape
            for x, y in [(w // 2, h // 2)] + ([click["pt"]] if click["pt"] else []):
                d = median_mm(depth, x, y)
                label(color, x, y, d)
                label(view_rgb, x, y, d)
            text(view_rgb, f"RGB  cam->PC {lat_rgb:.0f} ms" + ("   [overlay]" if overlay else ""), h - 12)
            text(color, f"{shown_fps:.0f} fps  cam->PC {lat_depth:.0f} ms  fill {100.0 * (depth > 0).mean():.0f}%",
                 h - 12)
            view = np.hstack([view_rgb, color])
            cv2.imshow(win, view)
            if args.peek_every and now - last_peek >= args.peek_every:
                last_peek = now
                peek_dir.mkdir(exist_ok=True)
                valid = depth[depth > 0]
                write_atomic(peek_dir / "rgbd_view.jpg", lambda f: cv2.imwrite(f, view, [cv2.IMWRITE_JPEG_QUALITY, 90]))
                write_atomic(peek_dir / "rgbd_stats.json", lambda f: Path(f).write_text(json.dumps({
                    "time": time.strftime("%H:%M:%S"), "fps": round(shown_fps, 1),
                    "latency_ms": {"rgb": round(lat_rgb), "depth": round(lat_depth)},
                    "fill_pct": round(100.0 * valid.size / depth.size, 1),
                    "depth_mm_p5_p50_p95": [int(v) for v in np.percentile(valid, [5, 50, 95])] if valid.size else None,
                    "center_mm": median_mm(depth, w // 2, h // 2),
                }, indent=1)))
        else:
            time.sleep(0.002)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_AUTOSIZE) < 0:
            break
        if key == ord("o"):
            overlay = not overlay
        if key == ord("s"):
            snap_dir.mkdir(exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(snap_dir / f"{stamp}_rgb.png"), frame)
            cv2.imwrite(str(snap_dir / f"{stamp}_depth_mm.png"), depth)
            print(f"Saved snapshot {stamp} to {snap_dir}", flush=True)
