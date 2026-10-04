"""Live stereo depth from the OV9282 pair on the OAK-FFC-3P, using the calibration stored on the device.

Left: rectified CAM_B image. Right: colorized depth (near = red, far = blue, black = no depth).
The center crosshair shows the median distance in a small box; click anywhere to measure that point.

Usage:
    python depth_view.py                     # 1280x800, HIGH_DETAIL, no decimation
    python depth_view.py --res 400           # 640x400 (faster, lower detail)
    python depth_view.py --near              # extended disparity: halves the minimum distance
    python depth_view.py --preset ROBOTICS
Press 'q' or Esc to quit.
"""
import argparse

import cv2
import depthai as dai
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--res", type=int, default=800, choices=[400, 800], help="Camera height: 800 = 1280x800, 400 = 640x400")
parser.add_argument("--fps", type=float, default=None, help="Default: picked from USB speed and resolution")
parser.add_argument("--preset", default="HIGH_DETAIL",
                    choices=["DEFAULT", "ACCURACY", "DENSITY", "ROBOTICS", "HIGH_DETAIL", "FAST_ACCURACY", "FAST_DENSITY"])
parser.add_argument("--near", action="store_true", help="Extended disparity, for objects closer than ~60 cm at 1280x800")
parser.add_argument("--no-temporal", action="store_true", help="Disable temporal filter (use when things move fast)")
parser.add_argument("--max-m", type=float, default=4.0, help="Distance mapped to the far end of the color scale")
args = parser.parse_args()

SIZE = (1280, 800) if args.res == 800 else (640, 400)
DISPLAY_SCALE = 0.5 if args.res == 800 else 1.0  # keep the side-by-side window on a laptop screen
click = {"pt": None}


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        half = int(SIZE[0] * DISPLAY_SCALE)
        click["pt"] = (int((x % half) / DISPLAY_SCALE), int(y / DISPLAY_SCALE))  # either half maps to the same pixel


def median_mm(depth, x, y, r=7):
    patch = depth[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
    valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size else None


def label(img, x, y, mm):
    text = f"{mm / 1000:.3f} m" if mm else "no depth"
    cv2.drawMarker(img, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
    for color, thick in (((0, 0, 0), 5), ((255, 255, 255), 2)):
        cv2.putText(img, text, (x + 14, y - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thick)


with dai.Pipeline() as pipeline:
    device = pipeline.getDefaultDevice()
    usb = device.getUsbSpeed()
    fps = args.fps or (30 if args.res == 400 else (20 if usb in (dai.UsbSpeed.SUPER, dai.UsbSpeed.SUPER_PLUS) else 10))
    calib = device.readCalibration()
    left_cam, right_cam = dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C
    print(f"USB {usb.name}, {SIZE[0]}x{SIZE[1]} @ {fps} fps, preset {args.preset}")
    print(f"Baseline from device calibration: measured "
          f"{calib.getBaselineDistance(left_cam, right_cam, useSpecTranslation=False):.2f} cm, "
          f"spec {calib.getBaselineDistance(left_cam, right_cam, useSpecTranslation=True):.2f} cm")

    left = pipeline.create(dai.node.Camera).build(left_cam)
    right = pipeline.create(dai.node.Camera).build(right_cam)
    stereo = pipeline.create(dai.node.StereoDepth).build(
        left.requestOutput(SIZE, fps=fps),
        right.requestOutput(SIZE, fps=fps),
        presetMode=getattr(dai.node.StereoDepth.PresetMode, args.preset),
    )
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setExtendedDisparity(args.near)
    stereo.setRectifyEdgeFillColor(0)

    pp = stereo.initialConfig.postProcessing
    pp.decimationFilter.decimationFactor = 1  # keep full depth resolution
    pp.speckleFilter.enable = True
    pp.speckleFilter.speckleRange = 50
    pp.spatialFilter.enable = True
    pp.spatialFilter.holeFillingRadius = 2
    pp.spatialFilter.numIterations = 1
    pp.temporalFilter.enable = not args.no_temporal

    q_depth = stereo.depth.createOutputQueue(maxSize=4, blocking=False)
    q_rect = stereo.rectifiedLeft.createOutputQueue(maxSize=4, blocking=False)

    pipeline.start()
    win = "OAK-FFC-3P depth (rectified left | depth)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print("Streaming depth. Click to measure a point, 'q' to quit.")

    rect = np.zeros((SIZE[1], SIZE[0]), np.uint8)
    while pipeline.isRunning():
        frame = q_rect.tryGet()
        if frame is not None:
            rect = frame.getCvFrame()
        msg = q_depth.get()
        depth = msg.getFrame()  # uint16, millimetres
        if depth.shape[:2] != (SIZE[1], SIZE[0]):  # some presets decimate the depth output
            depth = cv2.resize(depth, SIZE, interpolation=cv2.INTER_NEAREST)

        valid = depth > 0
        norm = np.clip(depth.astype(np.float32) / (args.max_m * 1000.0), 0, 1)
        color = cv2.applyColorMap(((1.0 - norm) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        color[~valid] = 0

        rect_bgr = cv2.cvtColor(rect, cv2.COLOR_GRAY2BGR)
        cx, cy = SIZE[0] // 2, SIZE[1] // 2
        points = [(cx, cy)] + ([click["pt"]] if click["pt"] else [])
        for x, y in points:
            mm = median_mm(depth, x, y)
            label(color, x, y, mm)
            label(rect_bgr, x, y, mm)

        cv2.putText(color, f"fill {100.0 * valid.mean():.0f}%", (14, SIZE[1] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        view = np.hstack([rect_bgr, color])
        if DISPLAY_SCALE != 1.0:
            view = cv2.resize(view, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_AREA)
        cv2.imshow(win, view)

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
