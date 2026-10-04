"""Live side-by-side preview of the OV9282 stereo pair on the OAK-FFC-3P (CAM_B + CAM_C).

Usage:
    python preview_stereo.py            # 640x400 @ 30 fps (safe on USB 2)
    python preview_stereo.py --full     # 1280x800 @ 15 fps
Press 'q' or Esc in the window to quit.
"""
import argparse
import time

import cv2
import depthai as dai
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--full", action="store_true", help="Full 1280x800 resolution (lower fps)")
parser.add_argument("--fps", type=float, default=None, help="Override frame rate")
args = parser.parse_args()

size = (1280, 800) if args.full else (640, 400)
fps = args.fps or (15 if args.full else 30)
sockets = {"CAM_B": dai.CameraBoardSocket.CAM_B, "CAM_C": dai.CameraBoardSocket.CAM_C}

with dai.Pipeline() as pipeline:
    queues = {}
    for name, socket in sockets.items():
        cam = pipeline.create(dai.node.Camera).build(socket)
        out = cam.requestOutput(size, type=dai.ImgFrame.Type.GRAY8, fps=fps)
        queues[name] = out.createOutputQueue(maxSize=4, blocking=False)

    pipeline.start()
    print(f"Streaming CAM_B + CAM_C at {size[0]}x{size[1]} @ {fps} fps. Press 'q' to quit.")

    latest = {name: np.zeros((size[1], size[0]), np.uint8) for name in sockets}
    counts, t0, shown_fps = 0, time.monotonic(), 0.0
    while pipeline.isRunning():
        for name, q in queues.items():
            frame = q.tryGet()
            if frame is not None:
                latest[name] = frame.getCvFrame()
                if name == "CAM_B":
                    counts += 1
        now = time.monotonic()
        if now - t0 >= 1.0:
            shown_fps, counts, t0 = counts / (now - t0), 0, now

        tiles = []
        for name, img in latest.items():
            tile = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.putText(tile, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            tiles.append(tile)
        view = np.hstack(tiles)
        cv2.putText(view, f"{shown_fps:.1f} fps", (10, view.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("OAK-FFC-3P stereo (CAM_B | CAM_C)", view)

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
