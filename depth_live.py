"""Low-latency stereo depth from the OAK-FFC-3P (CAM_B + CAM_C), streamed to view_depth.py on Windows.

Runs inside WSL (Smart App Control blocks depthai on Windows). Depth is computed on the device at 640x400 (full
sensor FOV, 2x2 binned) and sent raw (uint16 mm) next to the rectified left image, which is MJPEG-encoded on the
device. Every queue keeps only the newest frame, so the viewer never lags behind the cameras.

    python depth_live.py --calib resources/calib.json    # use a calibration file instead of the one on the device
    python depth_live.py --record-seconds 20              # also record rectified left/right + device depth (lossless)
Options: --fps 30 (the device keeps up to ~34 fps at 640x400), --near (extended disparity, ~16 cm minimum),
--alpha 0..1 (rectification FOV), --temporal, --spatial.
The on-board BNO086 IMU (100 Hz, same clock as the frames) adds per-frame orientation, angular speed and an
estimated rotational motion blur in pixels; recordings get imu.csv and per-frame IMU fields in meta.json.
"""
import argparse
import collections
import datetime
import json
import math
import os
import queue
import socket
import struct
import threading
import time

os.environ.setdefault("DEPTHAI_BOOTUP_TIMEOUT", "60000")  # usbipd re-attaches the booted device after ~10 s
os.environ.setdefault("DEPTHAI_CONNECT_TIMEOUT", "60000")

import cv2
import depthai as dai
import numpy as np

HEADER = struct.Struct("<BIfIHH")  # kind (0 = left JPEG, 1 = depth uint16, 2 = IMU JSON), seq, latency ms, len, w, h
SIZE = (640, 400)

parser = argparse.ArgumentParser()
parser.add_argument("--fps", type=float, default=30)
parser.add_argument("--calib", default=None, help="Calibration JSON to use for this run (device EEPROM untouched)")
parser.add_argument("--preset", default="DEFAULT", choices=["DEFAULT", "ACCURACY", "DENSITY", "ROBOTICS",
                                                            "HIGH_DETAIL", "FAST_ACCURACY", "FAST_DENSITY"])
parser.add_argument("--near", action="store_true", help="Extended disparity: halves the minimum distance")
parser.add_argument("--alpha", type=float, default=None,
                    help="Rectification scaling: 0 crops to valid pixels, 1 keeps the whole FOV (black borders)")
parser.add_argument("--spatial", action="store_true",
                    help="Spatial hole-filling filter: fewer holes, but caps the device at ~24 fps (+350 ms lag)")
parser.add_argument("--decimate", type=int, default=1, choices=[1, 2, 3, 4], help="Depth decimation factor")
parser.add_argument("--temporal", action="store_true", help="Temporal filter: smoother, but adds lag on motion")
parser.add_argument("--conf", type=int, default=None,
                    help="Confidence threshold 0-255 (default 15); higher is stricter: fewer matches, more holes")
parser.add_argument("--lrc", type=int, default=None, help="Left-right check threshold in pixels (lower = stricter)")
parser.add_argument("--max-exposure-us", type=int, default=8000,
                    help="Auto-exposure limit; short exposures avoid motion blur on a moving head (0 = no limit)")
parser.add_argument("--quality", type=int, default=80, help="MJPEG quality of the rectified left image")
parser.add_argument("--port", type=int, default=5601)
parser.add_argument("--record-seconds", type=float, default=0,
                    help="Record this many seconds of rectified left/right PNGs + device depth (0 = off)")
parser.add_argument("--record-dir", default=os.path.expanduser("~/oakffc3p/recordings"))
parser.add_argument("--imu-rate", type=int, default=100, help="IMU rate in Hz (0 = IMU off)")
parser.add_argument("--blur-px", type=float, default=2.0, help="Rotational blur (px) above which a frame is flagged")
args = parser.parse_args()


class LatestSlots:
    """Keeps only the newest item per stream."""

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


def quat_to_euler_deg(qi, qj, qk, qw):
    """Roll/pitch/yaw in degrees of the IMU's game rotation vector (gravity-referenced, yaw relative to start)."""
    roll = math.atan2(2 * (qw * qi + qj * qk), 1 - 2 * (qi * qi + qj * qj))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qj - qk * qi))))
    yaw = math.atan2(2 * (qw * qk + qi * qj), 1 - 2 * (qj * qj + qk * qk))
    return [round(math.degrees(a), 2) for a in (roll, pitch, yaw)]


class ImuTracker:
    """BNO086 samples on the host clock: per-frame orientation and motion, optional CSV log while recording."""

    CSV_HEADER = "t_s,ax_mps2,ay_mps2,az_mps2,gx_radps,gy_radps,gz_radps,qi,qj,qk,qw\n"

    def __init__(self, focal_px):
        self.samples = collections.deque(maxlen=1000)  # (t, ax, ay, az, gx, gy, gz, qi, qj, qk, qw)
        self.focal_px, self.csv, self.csv_t0 = focal_px, None, 0.0

    def add(self, imu_data):
        for p in imu_data.packets:
            a, g, r = p.acceleroMeter, p.gyroscope, p.rotationVector
            sample = (g.getTimestamp().total_seconds(), a.x, a.y, a.z, g.x, g.y, g.z, r.i, r.j, r.k, r.real)
            self.samples.append(sample)
            if self.csv:
                self.csv.write(",".join(f"{v:.6f}" for v in (sample[0] - self.csv_t0,) + sample[1:]) + "\n")

    def at(self, t, exposure_s):
        """IMU state for a frame at time t: nearest sample, plus peak angular speed during the exposure and the
        rotational motion blur it implies (angular speed x exposure x focal length, ignores translation)."""
        if not self.samples:
            return None
        nearest = min(self.samples, key=lambda s: abs(s[0] - t))
        window = [s for s in self.samples if t - exposure_s - 0.01 <= s[0] <= t + 0.01] or [nearest]
        omega = max(math.sqrt(s[4] ** 2 + s[5] ** 2 + s[6] ** 2) for s in window)
        return {
            "quat_ijkw": [round(v, 6) for v in nearest[7:11]],
            "roll_pitch_yaw_deg": quat_to_euler_deg(*nearest[7:11]),
            "gyro_dps": round(math.degrees(omega), 1),
            "accel_mps2": [round(v, 3) for v in nearest[1:4]],
            "exposure_ms": round(exposure_s * 1000, 2),
            "blur_px": round(omega * exposure_s * self.focal_px, 2),
        }

    def start_csv(self, path, t0):
        self.csv_t0, self.csv = t0, open(path, "w")
        self.csv.write(self.CSV_HEADER)

    def stop_csv(self):
        if self.csv:
            self.csv.close()
            self.csv = None


def serve(slots, stop):
    server = socket.create_server(("0.0.0.0", args.port))
    print(f"Waiting for viewer on port {args.port}...", flush=True)
    conn, addr = server.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)
    print(f"Viewer connected from {addr[0]}", flush=True)
    slots.take(0)
    try:
        while not stop.is_set():
            for kind, (seq, latency, w, h, payload) in sorted(slots.take().items()):
                conn.sendall(HEADER.pack(kind, seq, latency, len(payload), w, h) + payload)
    except OSError:
        print("Viewer disconnected.", flush=True)
    finally:
        stop.set()
        conn.close()
        server.close()


class Recorder:
    """Writes synchronized rectified left/right + device depth frames (matched by sequence number) as PNGs."""

    def __init__(self, root, seconds, meta, imu=None):
        self.imu = imu
        self.dir = os.path.join(root, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        for sub in ("left", "right", "depth"):
            os.makedirs(os.path.join(self.dir, sub))
        self.seconds, self.meta, self.frames = seconds, meta, []
        self.pending, self.t0, self.t_first, self.done = {}, None, None, False
        self.jobs = queue.Queue()
        self.writer = threading.Thread(target=self._write, daemon=True)
        self.writer.start()

    def add(self, name, msg):
        if self.done:
            return
        seq = msg.getSequenceNum()
        entry = self.pending.setdefault(seq, {})
        entry[name] = msg
        if len(entry) < 3:
            return
        del self.pending[seq]
        for old in [s for s in self.pending if s < seq]:  # a stream dropped these, they can't be completed
            del self.pending[old]
        ts = entry["left"].getTimestamp().total_seconds()
        if self.t_first is None:
            self.t_first = ts
            print("REC waiting 1 s for auto-exposure to settle...", flush=True)
        if ts - self.t_first < 1.0:
            return
        if self.t0 is None:
            self.t0 = ts
            if self.imu:
                self.imu.start_csv(os.path.join(self.dir, "imu.csv"), ts)
            print(f"REC started -> {self.dir}", flush=True)
        if ts - self.t0 > self.seconds:
            self.finish()
            return
        idx = len(self.frames)
        frame = {"index": idx, "seq": seq, "t": round(ts - self.t0, 6)}
        if self.imu:
            frame["imu"] = self.imu.at(ts, entry["left"].getExposureTime().total_seconds())
        self.frames.append(frame)
        self.jobs.put((idx, entry["left"].getCvFrame(), entry["right"].getCvFrame(), entry["depth"].getFrame()))
        if idx % int(self.meta["fps"]) == 0:
            print(f"REC {ts - self.t0:5.1f}/{self.seconds:g} s, {idx} frames, write queue {self.jobs.qsize()}",
                  flush=True)

    def _write(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            idx, left, right, depth = job
            name = f"{idx:06d}.png"
            cv2.imwrite(os.path.join(self.dir, "left", name), left, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            cv2.imwrite(os.path.join(self.dir, "right", name), right, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            cv2.imwrite(os.path.join(self.dir, "depth", name), depth, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    def finish(self):
        if self.done:
            return
        self.done = True
        if self.imu:
            self.imu.stop_csv()
        self.jobs.put(None)
        self.writer.join()
        span = self.frames[-1]["t"] if self.frames else 0
        blurred = sum(1 for f in self.frames if (f.get("imu") or {}).get("blur_px", 0) > self.meta["blur_flag_px"])
        self.meta.update(frames=self.frames, num_frames=len(self.frames), frames_flagged_blurry=blurred,
                         measured_fps=round(len(self.frames) / span, 2) if span else 0)
        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=1)
        print(f"REC done: {len(self.frames)} frames ({self.meta['measured_fps']} fps, {blurred} flagged for motion "
              f"blur) saved to {self.dir}", flush=True)


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
with dai.Pipeline(device) as pipeline:
    calib = dai.CalibrationHandler(args.calib) if args.calib else device.readCalibration()
    if args.calib:
        pipeline.setCalibrationData(calib)
    B, C = dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C
    print(f"Calibration: {args.calib or 'device EEPROM'}, baseline "
          f"{calib.getBaselineDistance(B, C, useSpecTranslation=False):.3f} cm", flush=True)

    left = pipeline.create(dai.node.Camera).build(B)
    right = pipeline.create(dai.node.Camera).build(C)
    if args.max_exposure_us:
        for cam in (left, right):
            cam.initialControl.setAutoExposureLimit(args.max_exposure_us)
    stereo = pipeline.create(dai.node.StereoDepth).build(
        left.requestOutput(SIZE, fps=args.fps),
        right.requestOutput(SIZE, fps=args.fps),
        presetMode=getattr(dai.node.StereoDepth.PresetMode, args.preset),
    )
    stereo.setDepthAlign(dai.StereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setExtendedDisparity(args.near)
    stereo.setRectifyEdgeFillColor(0)
    if args.alpha is not None:
        stereo.setAlphaScaling(args.alpha)
    if args.conf is not None:
        stereo.initialConfig.setConfidenceThreshold(args.conf)
    if args.lrc is not None:
        stereo.initialConfig.setLeftRightCheckThreshold(args.lrc)

    pp = stereo.initialConfig.postProcessing
    # when recording, the device depth (only kept for comparison) is halved so left+right+depth fit the USB link
    pp.decimationFilter.decimationFactor = 2 if args.record_seconds and args.decimate == 1 else args.decimate
    pp.speckleFilter.enable = True
    pp.speckleFilter.speckleRange = 50
    pp.spatialFilter.enable = args.spatial
    pp.spatialFilter.holeFillingRadius = 2
    pp.spatialFilter.numIterations = 1
    pp.temporalFilter.enable = args.temporal
    pp.thresholdFilter.minRange = 100
    pp.thresholdFilter.maxRange = 15000

    imu_tracker, q_imu = None, None
    if args.imu_rate:
        imu = pipeline.create(dai.node.IMU)
        imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER, dai.IMUSensor.GYROSCOPE_CALIBRATED,
                             dai.IMUSensor.GAME_ROTATION_VECTOR], args.imu_rate)
        imu.setBatchReportThreshold(1)
        imu.setMaxBatchReports(20)
        q_imu = imu.out.createOutputQueue(maxSize=50, blocking=False)
        imu_tracker = ImuTracker(calib.getCameraIntrinsics(C, SIZE[0], SIZE[1])[0][0])

    def poll_imu():
        if q_imu is not None:
            for m in q_imu.tryGetAll():
                imu_tracker.add(m)

    def send_imu(frame_msg):
        if imu_tracker is not None:
            state = imu_tracker.at(frame_msg.getTimestamp().total_seconds(), frame_msg.getExposureTime().total_seconds())
            if state:
                state["blurry"] = state["blur_px"] > args.blur_px
                slots.put(2, (frame_msg.getSequenceNum(), 0.0, 0, 0, json.dumps(state).encode()))

    recorder = None
    if args.record_seconds:
        # raw rectified pair (lossless) for recording; the preview JPEG is encoded on the host from the same frames
        rec_queues = {"left": stereo.rectifiedLeft.createOutputQueue(maxSize=30, blocking=False),
                      "right": stereo.rectifiedRight.createOutputQueue(maxSize=30, blocking=False),
                      "depth": stereo.depth.createOutputQueue(maxSize=30, blocking=False)}
    else:
        manip = pipeline.create(dai.node.ImageManip)  # the encoder needs NV12, rectified output is RAW8
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)
        manip.setMaxOutputFrameSize(SIZE[0] * SIZE[1] * 3 // 2)
        stereo.rectifiedLeft.link(manip.inputImage)
        enc = pipeline.create(dai.node.VideoEncoder).build(
            manip.out, frameRate=args.fps, profile=dai.VideoEncoderProperties.Profile.MJPEG, quality=args.quality)
        queues = [(1, stereo.depth.createOutputQueue(maxSize=1, blocking=False)),
                  (0, enc.out.createOutputQueue(maxSize=1, blocking=False))]
    pipeline.start()
    if args.record_seconds:
        M2 = np.array(calib.getCameraIntrinsics(C, SIZE[0], SIZE[1]))  # rectified views use the right camera's K
        recorder = Recorder(args.record_dir, args.record_seconds, {
            "fps": args.fps, "width": SIZE[0], "height": SIZE[1],
            "baseline_mm": calib.getBaselineDistance(B, C, useSpecTranslation=False) * 10.0,
            "rectified_focal_px_from_calib": float(M2[0, 0]),
            "rectified_cx_cy_from_calib": [float(M2[0, 2]), float(M2[1, 2])],
            "depth_units": "uint16 mm, 0 = invalid", "images": "rectified grayscale, aligned to left",
            "max_exposure_us": args.max_exposure_us, "blur_flag_px": args.blur_px,
            "imu": {"sensor": "BNO086", "rate_hz": args.imu_rate, "file": "imu.csv",
                    "orientation": "game rotation vector (gravity-referenced, yaw relative to start, IMU frame)",
                    "note": "IMU-to-camera rotation not calibrated yet; angular speed and blur are frame-independent"}
            if args.imu_rate else None,
        }, imu_tracker)
        calib.eepromToJsonFile(os.path.join(recorder.dir, "calibration.json"))
    print(f"USB {device.getUsbSpeed().name}, depth {SIZE[0]}x{SIZE[1]} @ {args.fps:g} fps, preset {args.preset}, "
          f"near={args.near}, temporal={args.temporal}, max exposure {args.max_exposure_us} us, confidence {stereo.initialConfig.getConfidenceThreshold()}, "
          f"LR threshold {stereo.initialConfig.getLeftRightCheckThreshold()}", flush=True)

    slots, stop = LatestSlots(), threading.Event()
    threading.Thread(target=serve, args=(slots, stop), daemon=True).start()
    while recorder is not None and pipeline.isRunning() and not stop.is_set():
        got = False
        poll_imu()
        for name, q in rec_queues.items():
            for m in q.tryGetAll():
                got = True
                recorder.add(name, m)
                latency = (dai.Clock.now() - m.getTimestamp()).total_seconds() * 1000.0
                if name == "left":
                    ok, jpg = cv2.imencode(".jpg", m.getCvFrame(), [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                    slots.put(0, (m.getSequenceNum(), latency, SIZE[0], SIZE[1], jpg.tobytes()))
                    send_imu(m)
                elif name == "depth":
                    slots.put(1, (m.getSequenceNum(), latency, m.getWidth(), m.getHeight(), m.getData().tobytes()))
        if recorder.done:
            break
        if not got:
            time.sleep(0.0005)
    if recorder is not None:
        recorder.finish()
    while recorder is None and pipeline.isRunning() and not stop.is_set():
        got = False
        poll_imu()
        for kind, q in queues:
            msgs = q.tryGetAll()
            if msgs:
                m = msgs[-1]
                latency = (dai.Clock.now() - m.getTimestamp()).total_seconds() * 1000.0
                if kind == 1:
                    payload, w, h = m.getData().tobytes(), m.getWidth(), m.getHeight()
                else:
                    payload, w, h = m.getData().tobytes(), SIZE[0], SIZE[1]
                slots.put(kind, (m.getSequenceNum(), latency, w, h, payload))
                if kind == 0:
                    send_imu(m)
                got = True
        if not got:
            time.sleep(0.0005)
