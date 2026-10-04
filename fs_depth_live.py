"""FoundationStereo depth on the OAK-FFC-3P stereo pair (runs in WSL on the GPU).

The device rectifies CAM_B/CAM_C with its calibration (and computes its own stereo depth for comparison); the
rectified pair is sent raw and FoundationStereo (NVlabs, as used by projectaria_gen2_depth_from_stereo) estimates
disparity, converted to metric depth with depth = focal * baseline / disparity.

    python fs_depth_live.py --snapshot out.jpg      # one frame: left | device depth | FoundationStereo depth
    python fs_depth_live.py                          # live: serve to view_depth.py on Windows (port 5602)
Options: --model vits|vitl, --iters 8 (refinement iterations: fewer = faster), --scale 0.5 (inference size).
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
import torch

from fs_model import FoundationStereoRunner, colorize, disparity_to_mm

HEADER = struct.Struct("<BIfIHH")  # same protocol as depth_live.py: kind (0 = left JPEG, 1 = depth), seq, ms, len, w, h
SIZE = (640, 400)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="vits", choices=["vits", "vitl"], help="vits = faster, vitl = most accurate")
parser.add_argument("--iters", type=int, default=8, help="Refinement iterations (repo default 32)")
parser.add_argument("--scale", type=float, default=1.0, help="Inference resolution scale (0.5 = 320x200, faster)")
parser.add_argument("--fps", type=float, default=30)
parser.add_argument("--max-exposure-us", type=int, default=8000)
parser.add_argument("--snapshot", default=None, help="Write one comparison image here and exit")
parser.add_argument("--port", type=int, default=5602)
args = parser.parse_args()

runner = FoundationStereoRunner(args.model, args.iters, args.scale)


def infer_disparity(left, right):
    return runner.disparity(left, right)


device = None
for attempt in range(3):  # usbipd occasionally misses the first re-attach of the booted device
    try:
        device = dai.Device()
        break
    except RuntimeError as e:
        print(f"Device not reachable after boot ({e}), retrying...", flush=True)
        time.sleep(3)
if device is None:
    raise SystemExit("Could not open the OAK device")

with dai.Pipeline(device) as pipeline:
    calib = device.readCalibration()
    B, C = dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C
    cams = []
    for sock in (B, C):
        cam = pipeline.create(dai.node.Camera).build(sock)
        if args.max_exposure_us:
            cam.initialControl.setAutoExposureLimit(args.max_exposure_us)
        cams.append(cam)
    stereo = pipeline.create(dai.node.StereoDepth).build(
        cams[0].requestOutput(SIZE, fps=args.fps), cams[1].requestOutput(SIZE, fps=args.fps),
        presetMode=dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.setDepthAlign(dai.StereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setRectifyEdgeFillColor(0)
    stereo.initialConfig.postProcessing.decimationFilter.decimationFactor = 1
    q_left = stereo.rectifiedLeft.createOutputQueue(maxSize=1, blocking=False)
    q_right = stereo.rectifiedRight.createOutputQueue(maxSize=1, blocking=False)
    q_disp = stereo.disparity.createOutputQueue(maxSize=1, blocking=False)
    q_depth = stereo.depth.createOutputQueue(maxSize=1, blocking=False)
    pipeline.start()

    def latest_pair():
        """Newest rectified left/right frames with matching sequence numbers (+ device depth/disparity)."""
        frames = {}
        while True:
            for name, q in (("l", q_left), ("r", q_right), ("disp", q_disp), ("depth", q_depth)):
                m = q.tryGet()
                if m is not None:
                    frames.setdefault(m.getSequenceNum(), {})[name] = m
            ready = [s for s, f in frames.items() if "l" in f and "r" in f]
            if ready:
                f = frames[max(ready)]
                return f["l"], f["r"], f.get("depth"), f.get("disp")
            time.sleep(0.001)

    # focal * baseline in (disparity px * mm), measured from the device's own depth and disparity of the same frame:
    # this is exactly the rectified geometry the device used, so both depths are directly comparable
    fb = None
    while fb is None:
        l, r, d, s = latest_pair()
        if d is not None and s is not None and d.getSequenceNum() == s.getSequenceNum():
            depth_dev, disp_dev = d.getFrame().astype(np.float64), s.getFrame().astype(np.float64)
            disp_px = disp_dev / (2 ** stereo.initialConfig.getSubpixelFractionalBits())
            ok = (depth_dev > 0) & (disp_px > 1)
            if ok.sum() > 1000:
                fb = float(np.median(depth_dev[ok] * disp_px[ok]))
    baseline_mm = calib.getBaselineDistance(B, C, useSpecTranslation=False) * 10.0
    print(f"Rectified focal {fb / baseline_mm:.1f} px, baseline {baseline_mm:.2f} mm", flush=True)

    with torch.no_grad():  # warm-up (cudnn autotune, first-call allocations)
        l, r, _, _ = latest_pair()
        infer_disparity(l.getCvFrame(), r.getCvFrame())

    if args.snapshot:
        l, r, d, _ = latest_pair()
        t0 = time.perf_counter()
        disp = infer_disparity(l.getCvFrame(), r.getCvFrame())
        ms = (time.perf_counter() - t0) * 1000
        fs_mm = disparity_to_mm(disp, fb)
        dev_mm = d.getFrame() if d is not None else np.zeros((SIZE[1], SIZE[0]), np.uint16)
        tiles = [cv2.cvtColor(l.getCvFrame(), cv2.COLOR_GRAY2BGR), colorize(dev_mm), colorize(fs_mm)]
        for tile, text in zip(tiles, ["rectified left", f"device stereo (fill {100 * (dev_mm > 0).mean():.0f}%)",
                                      f"FoundationStereo {args.model} it{args.iters} ({ms:.0f} ms)"]):
            for c, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
                cv2.putText(tile, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, th)
        cv2.imwrite(args.snapshot, np.hstack(tiles))
        np.save(os.path.splitext(args.snapshot)[0] + "_fs_mm.npy", fs_mm)
        np.save(os.path.splitext(args.snapshot)[0] + "_dev_mm.npy", dev_mm)
        print(f"Snapshot written to {args.snapshot}, inference {ms:.0f} ms", flush=True)
        raise SystemExit(0)

    server = socket.create_server(("0.0.0.0", args.port))
    print(f"Waiting for viewer on port {args.port}...", flush=True)
    conn, addr = server.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"Viewer connected from {addr[0]}", flush=True)
    try:
        while pipeline.isRunning():
            l, r, _, _ = latest_pair()  # always the newest pair: latency = one inference, never a backlog
            disp = infer_disparity(l.getCvFrame(), r.getCvFrame())
            fs_mm = disparity_to_mm(disp, fb)
            latency = (dai.Clock.now() - l.getTimestamp()).total_seconds() * 1000.0
            ok, jpg = cv2.imencode(".jpg", l.getCvFrame(), [cv2.IMWRITE_JPEG_QUALITY, 85])
            seq = l.getSequenceNum()
            conn.sendall(HEADER.pack(0, seq, latency, len(jpg), SIZE[0], SIZE[1]) + jpg.tobytes())
            conn.sendall(HEADER.pack(1, seq, latency, fs_mm.nbytes, SIZE[0], SIZE[1]) + fs_mm.tobytes())
    except OSError:
        print("Viewer disconnected.", flush=True)
    finally:
        conn.close()
        server.close()
