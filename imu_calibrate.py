"""Camera-IMU calibration for the OAK-FFC-3P (BNO086 on the base board, CAM_B left camera) - a light Kalibr.

    python imu_calibrate.py capture               # guided recording (native Windows, USB2 is fine)
    python imu_calibrate.py solve [REC_DIR]       # estimate everything from the newest (or given) recording
    python imu_calibrate.py solve REC_DIR --flash # also write the IMU->camera extrinsics into the device EEPROM

Capture: left camera 640x400 @ 60 fps with a short exposure (sharp while moving) + raw accelerometer/gyroscope at
400 Hz, all on the device clock. Show the charuco board (charuco_11_8, 3.8 cm squares) on the monitor and follow the
prompts: hold still, then move the rig in front of the board (rotate about all three axes, keep the board in view),
then rest it in ~6 different orientations for 3 s each (screen up/down, each side down...).

Solve:
  * camera orientation/position per frame from the charuco board (solvePnP, calibration from the device)
  * time offset IMU->camera: cross-correlation of |angular velocity| from the camera and from the gyroscope
  * rotation IMU->camera and gyro bias: least squares on w_cam = R (w_imu - b_g)
  * accelerometer bias and per-axis scale: static poses must read |g|
  * IMU position in the camera frame (lever arm): linear least squares on the accelerometer vs the camera trajectory
Writes REC/imu_calib.json.
"""
import argparse
import datetime
import glob
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SIZE = (640, 400)
G = 9.80665
SQUARES, SQUARE_CM, MARKER_CM = (11, 8), 3.8, 2.85

parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest="cmd", required=True)
cap = sub.add_parser("capture")
cap.add_argument("--fps", type=float, default=60)
cap.add_argument("--exposure-us", type=int, default=2000, help="Auto-exposure limit (short = no motion blur)")
cap.add_argument("--imu-rate", type=int, default=400)
cap.add_argument("--seconds", type=float, default=0, help="Stop automatically after this long (0 = press q)")
cap.add_argument("--out", default=str(ROOT / "recordings"))
sol = sub.add_parser("solve")
sol.add_argument("recording", nargs="?", default=None)
sol.add_argument("--flash", action="store_true", help="Write IMU->CAM_B extrinsics to the device (backup first)")
sol.add_argument("--min-corners", type=int, default=12)
sol.add_argument("--lowpass-hz", type=float, default=2.5,
                 help="Low-pass for the lever-arm fit (hand motion is < 2 Hz; lower cuts pose noise, too low cuts signal)")
sol.add_argument("--lever-cm", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                 help="IMU position in the left camera frame measured by hand (x right, y down, z forward, cm): "
                      "used instead of the estimate, which is only good to ~1-2 cm")
args = parser.parse_args()

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
board = cv2.aruco.CharucoBoard_create(SQUARES[0], SQUARES[1], SQUARE_CM / 100, MARKER_CM / 100, aruco_dict)  # metres
aruco_params = cv2.aruco.DetectorParameters_create()
aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.25  # screen bloom / blur, see calibrate.py


def detect(gray):
    mc, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    if ids is None or len(ids) < 4:
        return None, None
    n, cc, ci = cv2.aruco.interpolateCornersCharuco(mc, ids, gray, board)
    return (cc, ci) if n else (None, None)


# ===================================================================================================== capture ====
def capture():
    import depthai as dai
    out = Path(args.out) / ("imu_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    (out / "left").mkdir(parents=True)
    device = dai.Device()
    device.readCalibration().eepromToJsonFile(str(out / "calibration.json"))
    B = dai.CameraBoardSocket.CAM_B
    with dai.Pipeline(device) as pipeline:
        cam = pipeline.create(dai.node.Camera).build(B)
        cam.initialControl.setAutoExposureLimit(args.exposure_us)
        q_left = cam.requestOutput(SIZE, type=dai.ImgFrame.Type.GRAY8, fps=args.fps).createOutputQueue(
            maxSize=120, blocking=False)
        imu = pipeline.create(dai.node.IMU)
        imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], args.imu_rate)
        imu.setBatchReportThreshold(10)
        imu.setMaxBatchReports(40)
        q_imu = imu.out.createOutputQueue(maxSize=200, blocking=False)
        pipeline.start()

        accel_f, gyro_f, frames_f = (open(out / n, "w") for n in ("imu_accel.csv", "imu_gyro.csv", "frames.csv"))
        accel_f.write("t_device_s,ax_mps2,ay_mps2,az_mps2\n")
        gyro_f.write("t_device_s,gx_radps,gy_radps,gz_radps\n")
        frames_f.write("index,t_device_s,exposure_us,corners\n")
        last_a = last_g = -1.0
        steps = [(10, "1/3  HOLD STILL (put it down on the table)"),
                 (60, "2/3  MOVE: rotate about all 3 axes + translate, keep the board in view"),
                 (30, "3/3  REST in ~6 orientations, 3 s each (each face/side down)")]
        win = "IMU calibration capture  (q = stop)"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        t0, index, seen = time.monotonic(), 0, 0
        try:
            while pipeline.isRunning():
                for msg in q_imu.tryGetAll():
                    for p in msg.packets:
                        a, g = p.acceleroMeter, p.gyroscope
                        ta, tg = a.getTimestampDevice().total_seconds(), g.getTimestampDevice().total_seconds()
                        if ta > last_a:
                            accel_f.write(f"{ta:.6f},{a.x:.6f},{a.y:.6f},{a.z:.6f}\n")
                            last_a = ta
                        if tg > last_g:
                            gyro_f.write(f"{tg:.6f},{g.x:.6f},{g.y:.6f},{g.z:.6f}\n")
                            last_g = tg
                frames = q_left.tryGetAll()
                if not frames:
                    time.sleep(0.002)
                for m in frames:
                    gray = m.getFrame()
                    cv2.imwrite(str(out / "left" / f"{index:06d}.png"), gray, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                    cc, ci = detect(gray)
                    n = 0 if ci is None else len(ci)
                    seen += n >= args_min_corners
                    frames_f.write(f"{index},{m.getTimestampDevice().total_seconds():.6f},"
                                   f"{int(m.getExposureTime().total_seconds() * 1e6)},{n}\n")
                    index += 1
                    if index % 3 == 0:
                        el = time.monotonic() - t0
                        acc, step = 0, steps[-1][1] + "  (q when done)"
                        for dur, text in steps:
                            if el < acc + dur:
                                step = f"{text}  {acc + dur - el:4.0f} s"
                                break
                            acc += dur
                        view = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                        if ci is not None:
                            cv2.aruco.drawDetectedCornersCharuco(view, cc, ci, (0, 255, 0))
                        for c, th in (((0, 0, 0), 4), ((0, 255, 255), 1)):
                            cv2.putText(view, step, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, th)
                            cv2.putText(view, f"{el:5.1f} s  corners {n:2d}  board frames {seen}", (8, SIZE[1] - 12),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, th)
                        cv2.imshow(win, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_AUTOSIZE) < 0:
                    break
                if args.seconds and time.monotonic() - t0 >= args.seconds:
                    break
        finally:
            for f in (accel_f, gyro_f, frames_f):
                f.close()
            cv2.destroyAllWindows()
            json.dump({"fps": args.fps, "size": SIZE, "exposure_limit_us": args.exposure_us, "imu_rate": args.imu_rate,
                       "frames": index, "board": {"squares": SQUARES, "square_cm": SQUARE_CM, "marker_cm": MARKER_CM}},
                      open(out / "meta.json", "w"), indent=1)
            print(f"Saved {index} frames ({seen} with the board) to {out}\nNext:  python imu_calibrate.py solve {out}")


args_min_corners = 12


# ======================================================================================================= solve ====
def rodrigues_log(R):
    return cv2.Rodrigues(R)[0].ravel()


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def solve():
    import depthai as dai
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    rec = args.recording or max(glob.glob(str(ROOT / "recordings" / "imu_*")), key=os.path.getmtime)
    rec = Path(rec)
    meta = json.load(open(rec / "meta.json"))
    calib = dai.CalibrationHandler(str(rec / "calibration.json"))
    B = dai.CameraBoardSocket.CAM_B
    K = np.array(calib.getCameraIntrinsics(B, *meta["size"]))
    D = np.array(calib.getDistortionCoefficients(B))
    acc = np.loadtxt(rec / "imu_accel.csv", delimiter=",", skiprows=1)
    gyr = np.loadtxt(rec / "imu_gyro.csv", delimiter=",", skiprows=1)
    frames = np.loadtxt(rec / "frames.csv", delimiter=",", skiprows=1)
    print(f"{rec.name}: {len(frames)} frames, {len(gyr)} gyro / {len(acc)} accel samples "
          f"({len(gyr) / (gyr[-1, 0] - gyr[0, 0]):.0f} Hz)", flush=True)

    # ---- camera poses from the board ----
    t_cam, R_bc, p_bc = [], [], []  # camera-to-board rotation and camera position in the board frame
    for idx, t, _, n in frames:
        if n < args.min_corners:
            continue
        gray = cv2.imread(str(rec / "left" / f"{int(idx):06d}.png"), cv2.IMREAD_GRAYSCALE)
        cc, ci = detect(gray)
        if ci is None or len(ci) < args.min_corners:
            continue
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(cc, ci, board, K, D, None, None)
        if not ok:
            continue
        Rcb = cv2.Rodrigues(rvec)[0]  # board -> camera
        t_cam.append(t)
        R_bc.append(Rcb.T)
        p_bc.append((-Rcb.T @ tvec).ravel())
    t_cam, R_bc, p_bc = np.array(t_cam), np.array(R_bc), np.array(p_bc)
    if len(t_cam) < 200:
        raise SystemExit(f"Only {len(t_cam)} frames with a usable board pose - record again with the board in view")
    # keep the longest-enough runs without gaps (finite differences need consecutive frames)
    dt_frame = 1.0 / meta["fps"]
    print(f"Board pose in {len(t_cam)} frames", flush=True)

    # ---- camera angular velocity (camera frame) from consecutive orientations ----
    w_cam, t_w = [], []
    for k in range(len(t_cam) - 1):
        dt = t_cam[k + 1] - t_cam[k]
        if dt > 2.5 * dt_frame:
            continue
        w_cam.append(rodrigues_log(R_bc[k].T @ R_bc[k + 1]) / dt)
        t_w.append(0.5 * (t_cam[k] + t_cam[k + 1]))
    w_cam, t_w = np.array(w_cam), np.array(t_w)

    # ---- static periods: gyro AND accelerometer steady for >= 1 s (independent of bias; an orientation change
    #      always splits a period, even if the rig was turned over quickly) ----
    imu_rate = len(gyr) / (gyr[-1, 0] - gyr[0, 0])
    win = int(round(imu_rate))  # 1 s

    def moving_std(x):
        k = np.ones(win) / win
        mean = np.stack([np.convolve(x[:, i], k, mode="same") for i in range(x.shape[1])], 1)
        sq = np.stack([np.convolve(x[:, i] ** 2, k, mode="same") for i in range(x.shape[1])], 1)
        return np.sqrt(np.maximum(sq - mean ** 2, 0)).max(1)

    acc_on_g = np.stack([np.interp(gyr[:, 0], acc[:, 0], acc[:, i + 1]) for i in range(3)], 1)
    static = (moving_std(gyr[:, 1:]) < 0.01) & (moving_std(acc_on_g) < 0.15)
    segments, start = [], None
    for i, s in enumerate(np.r_[static, False]):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if i - start > win:
                segments.append((start, i))
            start = None
    static_time = sum(e - s for s, e in segments) / imu_rate
    b_g0 = np.median(np.vstack([gyr[s:e, 1:] for s, e in segments]), 0) if segments else np.zeros(3)
    gyro_noise = float(np.mean([np.std(gyr[s:e, 1:], 0).mean() for s, e in segments])) if segments else None
    fix_bias = static_time >= 2.0
    print(f"Static: {len(segments)} periods, {static_time:.1f} s -> gyro bias {np.round(np.degrees(b_g0), 3).tolist()} deg/s"
          f"{' (held fixed)' if fix_bias else ' (too little still time: estimated with the rotation)'}", flush=True)

    kernel = max(1, int(round(imu_rate * dt_frame)))  # average the IMU over one frame interval
    g_s = np.stack([np.convolve(gyr[:, i + 1], np.ones(kernel) / kernel, mode="same") for i in range(3)], 1)

    def gyro_at(t, b, offset):
        return np.stack([np.interp(t + offset, gyr[:, 0], g_s[:, i]) for i in range(3)], 1) - b

    # ---- time offset from |w| cross-correlation ----
    offsets = np.arange(-0.15, 0.15, 0.0005)
    wc = np.linalg.norm(w_cam, axis=1)
    scores = [np.corrcoef(wc, np.linalg.norm(gyro_at(t_w, b_g0, o), axis=1))[0, 1] for o in offsets]
    off0 = offsets[int(np.argmax(scores))]
    print(f"Time offset (|w| correlation): {off0 * 1000:+.1f} ms (corr {max(scores):.3f})", flush=True)

    # ---- rotation IMU->camera: Kabsch, then robust refinement of rotation + time offset (+ bias if no still data) ----
    moving = np.linalg.norm(w_cam, axis=1) > 0.2
    wi = gyro_at(t_w, b_g0, off0)
    U, _, Vt = np.linalg.svd(wi[moving].T @ w_cam[moving])
    R0 = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T

    def unpack(p):
        return cv2.Rodrigues(p[:3])[0], (b_g0 if fix_bias else p[4:7]), p[3]

    def resid(p):
        R, b, o = unpack(p)
        return (w_cam[moving] - gyro_at(t_w[moving], b, o) @ R.T).ravel()

    p0 = np.r_[rodrigues_log(R0), off0] if fix_bias else np.r_[rodrigues_log(R0), off0, b_g0]
    sol = least_squares(resid, p0, loss="huber", f_scale=0.05, x_scale=[0.01] * 3 + [0.001] * (len(p0) - 3))
    R_ci, b_g, offset = unpack(sol.x)
    offset = float(offset)
    r = resid(sol.x).reshape(-1, 3)
    gyro_rms = float(np.sqrt(np.median(np.sum(r ** 2, 1)) / 3))  # robust: board-pose outliers dominate a plain RMS
    print(f"Rotation IMU->cam fitted on {moving.sum()} samples: residual {np.degrees(gyro_rms):.2f} deg/s (robust rms), "
          f"gyro bias {np.round(np.degrees(b_g), 3).tolist()} deg/s, time offset {offset * 1000:+.2f} ms", flush=True)

    # ---- accelerometer bias / scale from the static periods ----
    statics = np.array([np.mean(acc_on_g[s:e], 0) for s, e in segments])
    dirs = statics / np.linalg.norm(statics, axis=1, keepdims=True) if len(statics) else np.zeros((0, 3))
    coverage = int(sum(np.any(dirs @ v > 0.8) for v in np.vstack([np.eye(3), -np.eye(3)])))
    b_a, s_a = np.zeros(3), np.ones(3)
    if len(statics) >= 4 and coverage >= 3:
        # |S (a - b)| = g per pose, plus weak priors (scale 1 +/- 2 %, bias 0 +/- 0.5 m/s^2) that only matter along
        # directions the poses do not constrain - without them a missing orientation gives a degenerate fit
        def acc_resid(p):
            return np.r_[np.linalg.norm((statics - p[:3]) * p[3:], axis=1) - G, (p[3:] - 1) / 0.02 * 0.05, p[:3] / 0.5 * 0.05]

        sa = least_squares(acc_resid, np.r_[np.zeros(3), np.ones(3)])
        b_a, s_a = sa.x[:3], sa.x[3:]
        acc_note = f"fitted on {len(statics)} static poses ({coverage}/6 directions)" + \
                   ("" if coverage == 6 else " - rest it on every side for a complete fit")
    else:
        acc_note = f"not fitted: {len(statics)} static poses covering {coverage}/6 directions (need >= 4 poses, 3 dirs)"
    g_err = np.abs(np.linalg.norm((statics - b_a) * s_a, axis=1) - G) if len(statics) else np.array([np.nan])
    print(f"Accelerometer: {acc_note}; bias {np.round(b_a, 4).tolist()} m/s^2, scale {np.round(s_a, 4).tolist()}, "
          f"|g| error {np.mean(g_err):.4f} m/s^2", flush=True)

    # ---- lever arm. f_imu = R_ci^T [ R_bc^T (a_cam - g_b) + (alpha x + w x w x) p_c ]: linear in g_b and p_c.
    #      Board poses are differentiated twice, so every signal (trajectory, rates, accelerometer) goes through the
    #      same zero-phase 4 Hz low-pass first; otherwise mm-level pose noise becomes m/s^2 of acceleration noise ----
    from scipy.signal import butter, filtfilt
    bf, af = butter(4, args.lowpass_hz / (meta["fps"] / 2))
    lever, lever_note = None, "not estimated"
    keep = np.r_[True, np.diff(t_cam) < 2.5 * dt_frame]
    run_ids = np.cumsum(~keep)
    acc_frame = np.stack([np.convolve(acc[:, i + 1], np.ones(kernel) / kernel, mode="same") for i in range(3)], 1)
    rows_A, rows_b = [], []
    for rid in np.unique(run_ids):
        sel = run_ids == rid
        if sel.sum() < 3 * meta["fps"]:  # filtfilt needs a few seconds per run
            continue
        tt, pp, RR = t_cam[sel], p_bc[sel], R_bc[sel]
        p_f = filtfilt(bf, af, pp, axis=0)
        acc_b = np.gradient(np.gradient(p_f, tt, axis=0), tt, axis=0)
        rot = Rotation.from_matrix(RR)
        w = np.zeros((len(tt), 3))
        w[1:-1] = (rot[:-2].inv() * rot[2:]).as_rotvec() / (tt[2:] - tt[:-2])[:, None]
        w[0], w[-1] = w[1], w[-2]
        w_f = filtfilt(bf, af, w, axis=0)
        alpha = np.gradient(w_f, tt, axis=0)
        f_raw = np.stack([np.interp(tt + offset, acc[:, 0], acc_frame[:, i]) for i in range(3)], 1)
        f_f = filtfilt(bf, af, (f_raw - b_a) * s_a, axis=0)
        edge = int(meta["fps"])  # drop 1 s at both ends (filter transients)
        for k in range(edge, len(tt) - edge):
            Rt = R_ci.T @ RR[k].T
            # unknowns: gravity in the board frame, lever arm, residual accelerometer bias (IMU frame)
            rows_A.append(np.hstack([-Rt, R_ci.T @ (skew(alpha[k]) + skew(w_f[k]) @ skew(w_f[k])), np.eye(3)]))
            rows_b.append(f_f[k] - Rt @ acc_b[k])
    if len(rows_A) > 300:
        A_mat, b_vec = np.vstack(rows_A), np.concatenate(rows_b)
        x, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        g_b, p_c, db = x[:3], x[3:6], x[6:9]
        res = A_mat @ x - b_vec
        cov = np.linalg.inv(A_mat.T @ A_mat) * np.var(res)
        # the low-pass makes neighbouring rows correlated: scale the formal error by sqrt(rows per independent sample)
        p_std = np.sqrt(np.diag(cov))[3:6] * np.sqrt(meta["fps"] / (2 * args.lowpass_hz))
        lever = p_c
        lever_note = (f"{np.round(p_c * 100, 2).tolist()} cm (+/- {np.round(p_std * 100, 2).tolist()} cm), "
                      f"fitted |g| {np.linalg.norm(g_b):.3f} m/s^2, residual accel bias {np.round(db, 3).tolist()}, "
                      f"accel residual {np.sqrt(np.mean(res ** 2)):.3f} m/s^2")
    print(f"Lever arm (IMU position in the left camera frame): {lever_note}", flush=True)

    result = {
        "recording": str(rec), "camera": "CAM_B (left)",
        "R_cam_imu": R_ci.tolist(), "R_cam_imu_note": "w_cam = R_cam_imu @ (w_imu - gyro_bias)",
        "t_cam_imu_cm": None if lever is None else (lever * 100).tolist(),
        "time_offset_ms": offset * 1000, "time_offset_note": "IMU sample at t_imu matches camera frame at t_imu - offset",
        "gyro_bias_radps": b_g.tolist(), "gyro_residual_degps_rms": float(np.degrees(gyro_rms)),
        "accel_bias_mps2": b_a.tolist(), "accel_scale": s_a.tolist(), "accel_note": acc_note,
        "static_poses": int(len(statics)), "board_frames": int(len(t_cam)),
        "gyro_noise_radps_still": gyro_noise, "static_seconds": round(static_time, 1),
    }
    json.dump(result, open(rec / "imu_calib.json", "w"), indent=1)
    print(f"Saved {rec / 'imu_calib.json'}", flush=True)

    if args.flash:
        if args.lever_cm is not None:
            lever = np.array(args.lever_cm) / 100.0
            print(f"Using the measured lever arm {args.lever_cm} cm", flush=True)
        elif lever is None or np.any(np.abs(lever) > 0.3) or np.any(p_std > 0.02):
            raise SystemExit("Not flashing: the lever-arm estimate is missing or uncertain (> 2 cm). Measure the IMU "
                             "position relative to the left lens and pass it with --lever-cm X Y Z")
        with dai.Device() as device:
            current = device.readCalibration()
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = ROOT / "depthai-main" / "resources" / f"eeprom_backup_{device.getDeviceId()}_{stamp}.json"
            current.eepromToJsonFile(str(backup))
            current.setImuExtrinsics(B, R_ci.tolist(), (lever * 100).tolist(), (lever * 100).tolist())
            device.flashCalibration(current)
            print(f"Backed up EEPROM to {backup}; flashed IMU->CAM_B extrinsics", flush=True)


if args.cmd == "capture":
    capture()
else:
    solve()
