"""Record a synchronized RGB + stereo + IMU dataset from the OAK-FFC-3P for offline FoundationStereo processing.

Runs natively on Windows (or WSL). Everything is hardware-synchronized on the device (left/right within 0.02 ms, RGB
within 0.5 ms) and written without re-encoding:
    left/000000.png, right/000000.png   raw CAM_B / CAM_C, 1280x800 full resolution, 8-bit, lossless (not rectified)
    rgb/000000.jpg                      raw CAM_A, 2048x1536 (full sensor FOV), MJPEG from the device (quality 95)
    imu_accel.csv, imu_gyro.csv         BNO086 raw accelerometer (m/s^2) / gyroscope (rad/s), ~400 Hz, device clock
    frames.jsonl                        per frame: device timestamps, exposure and ISO of every camera
    calibration.json, meta.json         device calibration (EEPROM) used for processing, recording settings
Then run rgbd_process.py (FoundationStereo) on the folder.

USB2 carries ~27 MB/s: full-res stereo + RGB fits at 8 fps with no dropped frames.
    python rgbd_record.py                  # record until q / Esc (preview window shows left | rgb)
    python rgbd_record.py --seconds 20
Keys: q/Esc stop. Hold the rig steady or move slowly: motion blur costs more depth quality than anything else.
"""
import argparse
import datetime
import json
import os
import queue
import threading
import time
from datetime import timedelta
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

MONO_SIZE = (1280, 800)
RGB_SIZE = (2048, 1536)  # MJPEG needs width % 32 == 0; a uniform 1.0105x scale of the 2028x1520 calibration size

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=0, help="Stop after this many seconds (0 = until q)")
parser.add_argument("--fps", type=float, default=8, help="8 is the most USB2 sustains without dropping frames")
parser.add_argument("--quality", type=int, default=95, help="RGB MJPEG quality")
parser.add_argument("--max-exposure-us", type=int, default=8000,
                    help="Mono auto-exposure limit: short exposures avoid motion blur (0 = no limit)")
parser.add_argument("--rgb-max-exposure-us", type=int, default=0, help="RGB auto-exposure limit (0 = no limit)")
parser.add_argument("--imu-rate", type=int, default=400, help="Raw accelerometer/gyroscope rate in Hz (0 = off)")
parser.add_argument("--settle", type=float, default=1.5,
                    help="Minimum seconds before recording; it also waits (up to 8 s) until auto-exposure is stable")
parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "recordings"))
parser.add_argument("--png-level", type=int, default=1, help="PNG compression level 0-9 (lossless either way)")
args = parser.parse_args()

A, B, C = dai.CameraBoardSocket.CAM_A, dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C
out = Path(args.out) / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
for sub in ("left", "right", "rgb"):
    (out / sub).mkdir(parents=True)

jobs = queue.Queue(maxsize=256)


def writer():
    while True:
        job = jobs.get()
        if job is None:
            return
        path, data = job
        if isinstance(data, np.ndarray):
            cv2.imwrite(str(path), data, [cv2.IMWRITE_PNG_COMPRESSION, args.png_level])
        else:
            path.write_bytes(data)
        jobs.task_done()


writers = [threading.Thread(target=writer, daemon=True) for _ in range(4)]
for t in writers:
    t.start()

device = dai.Device()
calib = device.readCalibration()
calib.eepromToJsonFile(str(out / "calibration.json"))
usb = device.getUsbSpeed().name
print(f"Device {device.getDeviceId()} on USB {usb}, baseline "
      f"{calib.getBaselineDistance(C, B, useSpecTranslation=False):.3f} cm -> {out}", flush=True)

with dai.Pipeline(device) as pipeline:
    sync = pipeline.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(seconds=0.5 / args.fps))
    for sock, name in ((B, "left"), (C, "right")):
        cam = pipeline.create(dai.node.Camera).build(sock)
        if args.max_exposure_us:
            cam.initialControl.setAutoExposureLimit(args.max_exposure_us)
        cam.requestFullResolutionOutput(type=dai.ImgFrame.Type.GRAY8, fps=args.fps).link(sync.inputs[name])
    rgb = pipeline.create(dai.node.Camera).build(A)
    if args.rgb_max_exposure_us:
        rgb.initialControl.setAutoExposureLimit(args.rgb_max_exposure_us)
    enc = pipeline.create(dai.node.VideoEncoder).build(
        rgb.requestOutput(RGB_SIZE, type=dai.ImgFrame.Type.NV12, fps=args.fps),
        frameRate=args.fps, profile=dai.VideoEncoderProperties.Profile.MJPEG, quality=args.quality)
    enc.out.link(sync.inputs["rgb"])
    q_sync = sync.out.createOutputQueue(maxSize=64, blocking=False)
    q_imu = None
    if args.imu_rate:
        imu = pipeline.create(dai.node.IMU)
        imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], args.imu_rate)
        imu.setBatchReportThreshold(10)
        imu.setMaxBatchReports(40)
        q_imu = imu.out.createOutputQueue(maxSize=200, blocking=False)
    pipeline.start()

    accel_f = open(out / "imu_accel.csv", "w")
    gyro_f = open(out / "imu_gyro.csv", "w")
    accel_f.write("t_device_s,ax_mps2,ay_mps2,az_mps2\n")
    gyro_f.write("t_device_s,gx_radps,gy_radps,gz_radps\n")
    frames_f = open(out / "frames.jsonl", "w")
    last_accel = last_gyro = -1.0

    win = "Recording: left | rgb   (q = stop)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    t_start, t_rec, index, last_seq, dropped = time.monotonic(), None, 0, None, 0
    ev_history = []
    preview_every, last_preview = 1.0 / 8, 0.0
    try:
        while pipeline.isRunning():
            if q_imu is not None:
                for msg in q_imu.tryGetAll():
                    for p in msg.packets:
                        a, g = p.acceleroMeter, p.gyroscope
                        ta, tg = a.getTimestampDevice().total_seconds(), g.getTimestampDevice().total_seconds()
                        if t_rec is not None and ta > last_accel:  # packets repeat the latest sample of the slower sensor
                            accel_f.write(f"{ta:.6f},{a.x:.6f},{a.y:.6f},{a.z:.6f}\n")
                            last_accel = ta
                        if t_rec is not None and tg > last_gyro:
                            gyro_f.write(f"{tg:.6f},{g.x:.6f},{g.y:.6f},{g.z:.6f}\n")
                            last_gyro = tg
            groups = q_sync.tryGetAll()
            if not groups:
                time.sleep(0.002)
            for group in groups:
                left, right, enc_rgb = group["left"], group["right"], group["rgb"]
                now = time.monotonic()
                if t_rec is None:
                    # wait until every camera's exposure x gain has stopped changing (auto-exposure converged)
                    ev = [m.getExposureTime().total_seconds() * m.getSensitivity() for m in (left, right, enc_rgb)]
                    ev_history.append(ev)
                    recent = np.array(ev_history[-int(args.fps):])
                    stable = len(recent) >= args.fps and np.all(recent.max(0) <= 1.1 * recent.min(0))
                    if now - t_start < args.settle or (not stable and now - t_start < 8):
                        continue
                    t_rec = now
                    print("Recording... (q to stop)", flush=True)
                seq = left.getSequenceNum()
                if last_seq is not None and seq - last_seq > 1:
                    dropped += seq - last_seq - 1
                last_seq = seq
                name = f"{index:06d}"
                jobs.put((out / "left" / f"{name}.png", left.getFrame()))
                jobs.put((out / "right" / f"{name}.png", right.getFrame()))
                jobs.put((out / "rgb" / f"{name}.jpg", bytes(enc_rgb.getData())))
                info = {"index": index, "seq": seq}
                for key, m in (("left", left), ("right", right), ("rgb", enc_rgb)):
                    info[key] = {"t_device_s": round(m.getTimestampDevice().total_seconds(), 6),
                                 "exposure_us": int(m.getExposureTime().total_seconds() * 1e6),
                                 "iso": int(m.getSensitivity())}
                frames_f.write(json.dumps(info) + "\n")
                index += 1
                if now - last_preview >= preview_every:
                    last_preview = now
                    small_rgb = cv2.imdecode(np.frombuffer(enc_rgb.getData(), np.uint8), cv2.IMREAD_REDUCED_COLOR_4)
                    small_left = cv2.cvtColor(cv2.resize(left.getFrame(), (512, 320), interpolation=cv2.INTER_AREA),
                                              cv2.COLOR_GRAY2BGR)
                    small_left = cv2.copyMakeBorder(small_left, 32, 32, 0, 0, cv2.BORDER_CONSTANT)
                    view = np.hstack([small_left, small_rgb])
                    txt = (f"REC {now - t_rec:5.1f} s  {index} frames  dropped {dropped}  "
                           f"exp {info['left']['exposure_us']} us  writer queue {jobs.qsize()}")
                    for c, th in (((0, 0, 0), 4), ((0, 0, 255), 1)):
                        cv2.putText(view, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, th)
                    cv2.imshow(win, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_AUTOSIZE) < 0:
                break
            if args.seconds and t_rec is not None and time.monotonic() - t_rec >= args.seconds:
                break
    finally:
        duration = time.monotonic() - t_rec if t_rec else 0.0
        jobs.join()
        for _ in writers:
            jobs.put(None)
        for f in (accel_f, gyro_f, frames_f):
            f.close()
        cv2.destroyAllWindows()
        meta = {
            "device_id": device.getDeviceId(), "usb": usb, "depthai": dai.__version__,
            "fps": args.fps, "frames": index, "dropped": dropped, "duration_s": round(duration, 2),
            "measured_fps": round(index / duration, 2) if duration else None,
            "mono_size": MONO_SIZE, "rgb_size": RGB_SIZE, "rgb_quality": args.quality,
            "max_exposure_us": args.max_exposure_us, "rgb_max_exposure_us": args.rgb_max_exposure_us,
            "imu_rate_hz": args.imu_rate, "sockets": {"left": "CAM_B", "right": "CAM_C", "rgb": "CAM_A"},
            "baseline_cm": calib.getBaselineDistance(C, B, useSpecTranslation=False),
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"Saved {index} frames ({dropped} dropped) in {duration:.1f} s to {out}", flush=True)
