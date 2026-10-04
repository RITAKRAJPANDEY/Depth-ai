"""Offline FoundationStereo RGB-D processing for recordings made with rgbd_record.py (runs in WSL on the GPU).

Per frame:
  1. Rectify the raw 1280x800 left/right pair on the host in float precision from the calibration in the recording
     (14-coefficient rational distortion, OpenCV stereoRectify) - exact focal length and baseline, no device mesh.
  2. FoundationStereo (ViT-L, 32 refinement iterations, as in projectaria_gen2_depth_from_stereo) at full resolution,
     plus its left-right consistency check (second pass on the mirrored, swapped pair; inconsistent pixels dropped).
  3. depth = focal * baseline / disparity in the rectified left camera.
  4. Re-project the depth into the undistorted RGB camera (2048x1536) with a z-buffer (nearest surface wins).

Writes REC/processed/:
    rgb/000000.png         undistorted RGB (pinhole, intrinsics in cameras.json)
    depth/000000.png       uint16 depth in mm aligned pixel-for-pixel to rgb/ (0 = no depth)
    left_rect/000000.png   rectified left image (8-bit)      depth_left/000000.png  uint16 mm aligned to left_rect/
    disp_left/000000.png   raw FoundationStereo disparity, uint16 in 1/64 px (depth_mm = baseline*focal / disparity)
    lr_diff/000000.png     left-right disagreement, uint16 in 1/100 px (65535 = no match): re-threshold offline
    cameras.json           pinhole intrinsics of rgb and left_rect, baseline, T_rgb_leftrect, distortion-free
    frames.json            per frame timestamps / exposures + depth fill
    preview.avi            rgb | aligned depth | overlay        clouds/000000.ply  colored point clouds
    summary.json           timing, fill, rectification row error (calibration check)

    python rgbd_process.py REC_DIR                       # ViT-L, 32 iterations, LR check, every frame
    python rgbd_process.py REC_DIR --stride 4 --model vits --iters 16 --no-lr      # quick look
"""
import argparse
import glob
import json
import os
import time

import cv2
import depthai as dai
import numpy as np

from fs_model import FoundationStereoRunner, colorize

parser = argparse.ArgumentParser()
parser.add_argument("recording", help="Folder written by rgbd_record.py")
parser.add_argument("--model", default="vitl", choices=["vits", "vitl"])
parser.add_argument("--iters", type=int, default=32, help="Refinement iterations (Aria pipeline uses 32)")
parser.add_argument("--scale", type=float, default=1.0, help="Inference scale (1.0 = full 1280x800)")
parser.add_argument("--no-lr", action="store_true", help="Skip the left-right consistency pass (2x faster)")
parser.add_argument("--lr-threshold", type=float, default=1.0, help="Max left-right disparity disagreement (px)")
parser.add_argument("--no-refine", action="store_true",
                    help="Use the calibration as-is (skip the per-recording pitch/roll refinement from feature matches)")
parser.add_argument("--alpha", type=float, default=0.5,
                    help="Rectification scaling: 0 = only valid pixels (zooms in ~10%%, i.e. upsamples), 1 = whole FOV. "
                         "0.5 keeps the native ~900 px focal length: 71x48 deg, 93%% valid pixels")
parser.add_argument("--min-m", type=float, default=0.2)
parser.add_argument("--max-m", type=float, default=10.0, help="Depth beyond this is dropped (precision falls as Z^2)")
parser.add_argument("--stride", type=int, default=1)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--max-frames", type=int, default=0)
parser.add_argument("--ply-every", type=int, default=0, help="Also write a point cloud every N frames (0 = first only)")
parser.add_argument("--out", default=None, help="Output folder (default REC/processed)")
parser.add_argument("--resume", action="store_true",
                    help="Keep frames an interrupted run already finished; only process the missing ones")
parser.add_argument("--geometry-only", action="store_true", help="Only estimate/report the per-frame geometry")
args = parser.parse_args()

rec = os.path.abspath(args.recording)
out = args.out or os.path.join(rec, "processed")
for sub in ("rgb", "depth", "left_rect", "depth_left", "disp_left", "lr_diff", "clouds"):
    os.makedirs(os.path.join(out, sub), exist_ok=True)
meta = json.load(open(os.path.join(rec, "meta.json")))
frames_meta = [json.loads(l) for l in open(os.path.join(rec, "frames.jsonl"))]
names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(rec, "left", "*.png")))
names = names[args.start::args.stride]
if args.max_frames:
    names = names[:args.max_frames]

# ---- geometry from the recording's calibration -------------------------------------------------------------------
calib = dai.CalibrationHandler(os.path.join(rec, "calibration.json"))
A, B, C = dai.CameraBoardSocket.CAM_A, dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C
mono_size, rgb_size = tuple(meta["mono_size"]), tuple(meta["rgb_size"])


def intrinsics(sock, size):
    return (np.array(calib.getCameraIntrinsics(sock, size[0], size[1]), np.float64),
            np.array(calib.getDistortionCoefficients(sock), np.float64))


def extrinsics(src, dst):
    """X_dst = R @ X_src + t (t in mm)."""
    M = np.array(calib.getCameraExtrinsics(src, dst, useSpecTranslation=False), np.float64)
    return M[:3, :3], M[:3, 3] * 10.0


K_l, D_l = intrinsics(B, mono_size)
K_r, D_r = intrinsics(C, mono_size)
K_a, D_a = intrinsics(A, rgb_size)
R_lr, t_lr = extrinsics(B, C)
R_la, t_la = extrinsics(B, A)


# ---- extrinsic refinement -------------------------------------------------------------------------------------------
# The cameras on this FFC rig are not perfectly rigid: the right module was measured moving by ~0.5 deg in pitch AND
# yaw between recordings and even within one. Pitch/roll show up as vertical misalignment of the rectified rows (stereo
# matching degrades); yaw shows up as a disparity offset (every depth biased, ~2 % per 0.1 deg at 1 m). So:
#   * left->rgb pitch/roll: refined once per recording (the RGB has stayed stable)
#   * left->right pitch, roll AND yaw: estimated for every frame - pitch/roll from left-right feature matches, yaw from
#     the left-right disparity of points triangulated with the left-rgb pair (the third camera provides the scale) -
#     then smoothed over time (rolling median over ~1 s)
# The left rectification stays fixed, so left_rect/ images, their intrinsics and T_rgb_leftrect are constant.
from scipy.optimize import least_squares

SIFT = cv2.SIFT_create(4000)
MATCHER = cv2.BFMatcher()
RGB_GEOM_SCALE = 0.5  # SIFT on the half-size RGB: plenty for geometry, 4x faster


def features(img):
    k, d = SIFT.detectAndCompute(img, None)
    return np.float32([p.pt for p in k]).reshape(-1, 2), d


def match(d1, d2):
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return {}
    return {a.queryIdx: a.trainIdx for a, b in MATCHER.knnMatch(d1, d2, k=2) if a.distance < 0.7 * b.distance}


def normalized(pts, K, D):
    return cv2.undistortPoints(pts.reshape(-1, 1, 2), K, D)[:, 0] if len(pts) else np.zeros((0, 2), np.float32)


def sampson_px(x1, x2, R, t, f):
    """Sampson epipolar distance in pixels (focal f) for normalized points x1 (cam 1) and x2 (cam 2)."""
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]]) / np.linalg.norm(t)
    E = tx @ R
    h1, h2 = np.c_[x1, np.ones(len(x1))], np.c_[x2, np.ones(len(x2))]
    Ex1, Etx2 = h1 @ E.T, h2 @ E
    num = np.sum(h2 * Ex1, axis=1)
    return f * num / np.sqrt(Ex1[:, 0] ** 2 + Ex1[:, 1] ** 2 + Etx2[:, 0] ** 2 + Etx2[:, 1] ** 2)


def small_rotation(p):
    return cv2.Rodrigues(np.asarray(p, np.float64))[0]


# geometry pass: features of every frame (left shared between the left-right and left-rgb matches)
K_a_geom = K_a.copy()
K_a_geom[:2, :2] *= RGB_GEOM_SCALE
K_a_geom[:2, 2] = (K_a[:2, 2] + 0.5) * RGB_GEOM_SCALE - 0.5
json_frames = {f"{f['index']:06d}": f for f in frames_meta}
geo = []
if not args.no_refine:
    t_geo = time.perf_counter()
    for i, name in enumerate(names):
        L = cv2.imread(os.path.join(rec, "left", name + ".png"), cv2.IMREAD_GRAYSCALE)
        Rimg = cv2.imread(os.path.join(rec, "right", name + ".png"), cv2.IMREAD_GRAYSCALE)
        Aimg = cv2.resize(cv2.imread(os.path.join(rec, "rgb", name + ".jpg"), cv2.IMREAD_GRAYSCALE), None,
                          fx=RGB_GEOM_SCALE, fy=RGB_GEOM_SCALE, interpolation=cv2.INTER_AREA)
        (pl, dl), (pr, dr), (pa, da) = features(L), features(Rimg), features(Aimg)
        mr, ma = match(dl, dr), match(dl, da)
        nl, nr, na = normalized(pl, K_l, D_l), normalized(pr, K_r, D_r), normalized(pa, K_a_geom, D_a)
        ql, qa = list(ma), list(ma.values())
        common = sorted(set(mr) & set(ma))
        geo.append({"lr": (nl[list(mr)], nr[list(mr.values())]), "la": (nl[ql], na[qa]),
                    "c": (nl[common], nr[[mr[q] for q in common]], na[[ma[q] for q in common]])})
        if (i + 1) % 50 == 0 or i + 1 == len(names):
            print(f"  geometry features {i + 1}/{len(names)} ({(time.perf_counter() - t_geo) / (i + 1):.2f} s/frame)",
                  flush=True)

refinement = {}
# left->rgb pitch/roll, once per recording
if geo:
    x1 = np.concatenate([g["la"][0] for g in geo])
    x2 = np.concatenate([g["la"][1] for g in geo])
    if len(x1) > 200:
        before = np.abs(sampson_px(x1, x2, R_la, t_la, K_a[0, 0]))
        keep = before < 30
        sol = least_squares(lambda p: sampson_px(x1[keep], x2[keep], small_rotation([p[0], 0, p[1]]) @ R_la,
                                                 small_rotation([p[0], 0, p[1]]) @ t_la, K_a[0, 0]),
                            [0.0, 0.0], loss="huber", f_scale=1.0)
        dR = small_rotation([sol.x[0], 0, sol.x[1]])
        after = np.abs(sampson_px(x1[keep], x2[keep], dR @ R_la, dR @ t_la, K_a[0, 0]))
        b, a = float(np.median(before[keep])), float(np.median(after))
        if a < b:
            R_la, t_la = dR @ R_la, dR @ t_la
        refinement["left->rgb"] = {"pitch_deg": round(float(np.degrees(sol.x[0])), 4),
                                   "roll_deg": round(float(np.degrees(sol.x[1])), 4), "epipolar_px_before": round(b, 3),
                                   "epipolar_px_after": round(a, 3), "matches": int(keep.sum()), "applied": a < b}
        print(f"Refine left->rgb: pitch {np.degrees(sol.x[0]):+.3f} deg, roll {np.degrees(sol.x[1]):+.3f} deg, "
              f"epipolar {b:.2f} -> {a:.2f} px ({keep.sum()} matches)", flush=True)

# fixed left rectification from the calibration; the right camera's rectifying rotation follows it per frame
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K_l, D_l, K_r, D_r, mono_size, R_lr, t_lr.reshape(3, 1),
                                                  flags=cv2.CALIB_ZERO_DISPARITY, alpha=args.alpha)
map_l = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, mono_size, cv2.CV_32FC1)
f_rect, cx_rect, cy_rect = P1[0, 0], P1[0, 2], P1[1, 2]
baseline_mm = -P2[0, 3] / P2[0, 0]
fb = f_rect * baseline_mm
valid_l = cv2.remap(np.full(mono_size[::-1], 255, np.uint8), *map_l, cv2.INTER_NEAREST, borderValue=0) == 255


def rect_px(n, R):
    v = np.c_[n, np.ones(len(n))] @ R.T
    return np.c_[f_rect * v[:, 0] / v[:, 2] + cx_rect, P1[1, 1] * v[:, 1] / v[:, 2] + cy_rect]


def right_rect_rotation(p):
    """Rectifying rotation of the right camera for a small rotation p (rotation vector, right camera frame)."""
    return R1 @ (small_rotation(p) @ R_lr).T


per_frame = np.full((len(names), 3), np.nan)
frame_geo_info = [{} for _ in names]
if geo:
    p_prev = np.zeros(3)
    for k, g in enumerate(geo):
        nl_lr, nr_lr = g["lr"]
        nl_c, nr_c, na_c = g["c"]
        if len(nl_lr) < 30:
            continue
        rl = rect_px(nl_lr, R1)
        # 3-D points of the common features from the left-rgb pair (left camera frame, mm)
        X = cv2.triangulatePoints(np.eye(3, 4), np.c_[R_la, t_la], nl_c.T.astype(np.float64),
                                  na_c.T.astype(np.float64)) if len(nl_c) else np.zeros((4, 0))
        X = (X[:3] / X[3]).T if len(nl_c) else np.zeros((0, 3))
        Xa = X @ R_la.T + t_la
        good = (X[:, 2] > 150) & (X[:, 2] < 6000) & (Xa[:, 2] > 150)
        if good.any():
            good &= np.linalg.norm(Xa[:, :2] / Xa[:, 2:] - na_c, axis=1) * K_a[0, 0] < 1.5
        X, nl_c, nr_c = X[good], nl_c[good], nr_c[good]
        rl_c = rect_px(nl_c, R1)
        d_expected = fb / (X @ R1.T)[:, 2]
        use_yaw = len(X) >= 15

        def residuals(p, sel_v=slice(None), sel_h=slice(None)):
            R2k = right_rect_rotation(p if use_yaw else [p[0], p_prev[1], p[1]])
            r_v = rl[sel_v, 1] - rect_px(nr_lr[sel_v], R2k)[:, 1]
            if not use_yaw:
                return r_v
            r_h = (rl_c[sel_h, 0] - rect_px(nr_c[sel_h], R2k)[:, 0]) - d_expected[sel_h]
            return np.r_[r_v, r_h]

        x0 = p_prev if use_yaw else p_prev[[0, 2]]
        r0 = residuals(x0)
        n_v = len(nl_lr)
        ok_v = np.abs(r0[:n_v] - np.median(r0[:n_v])) < 10  # gross mismatches out
        ok_h = np.abs(r0[n_v:] - np.median(r0[n_v:])) < 10 if use_yaw else slice(None)
        sol = least_squares(lambda p: residuals(p, ok_v, ok_h), x0, loss="huber", f_scale=1.0)
        p = sol.x if use_yaw else np.array([sol.x[0], np.nan, sol.x[1]])
        per_frame[k] = p
        r = residuals(sol.x, ok_v, ok_h)
        frame_geo_info[k] = {"lr_matches": int(ok_v.sum()), "yaw_points": int(np.sum(ok_h)) if use_yaw else 0,
                             "row_residual_px": round(float(np.median(np.abs(r[:ok_v.sum()]))), 3)}
        p_prev = np.where(np.isfinite(p), p, p_prev)

    # temporal smoothing: rolling median over +/- 0.5 s of recording time, gaps filled by interpolation
    t_frames = np.array([json_frames.get(n, {}).get("left", {}).get("t_device_s", float(n) / meta["fps"])
                         for n in names])
    smooth = np.full_like(per_frame, np.nan)
    for k in range(len(names)):
        window = per_frame[np.abs(t_frames - t_frames[k]) <= 0.5]
        for j in range(3):
            vals = window[:, j][np.isfinite(window[:, j])]
            if len(vals):
                smooth[k, j] = np.median(vals)
    for j in range(3):
        ok = np.isfinite(smooth[:, j])
        smooth[:, j] = np.interp(t_frames, t_frames[ok], smooth[ok, j]) if ok.any() else 0.0
    if not np.isfinite(per_frame[:, 1]).any():
        print("WARNING: no left-rgb-right common features - yaw could not be estimated (depth may be biased)",
              flush=True)
    per_frame_smooth = smooth
    deg = np.degrees(smooth)
    refinement["left->right_per_frame"] = {
        k: {"median_deg": round(float(np.median(deg[:, j])), 4), "min_deg": round(float(deg[:, j].min()), 4),
            "max_deg": round(float(deg[:, j].max()), 4)} for j, k in enumerate(("pitch", "yaw", "roll"))}
    refinement["left->right_per_frame"]["frames_with_yaw"] = int(np.isfinite(per_frame[:, 1]).sum())
    print("Refine left->right per frame: " + ", ".join(
        f"{k} {deg[:, j].min():+.3f}..{deg[:, j].max():+.3f} deg" for j, k in enumerate(("pitch", "yaw", "roll"))) +
          f" (yaw measured in {np.isfinite(per_frame[:, 1]).sum()}/{len(names)} frames)", flush=True)
else:
    per_frame_smooth = np.zeros((len(names), 3))

K_rgb, _ = cv2.getOptimalNewCameraMatrix(K_a, D_a, rgb_size, alpha=0, newImgSize=rgb_size)
map_a = cv2.initUndistortRectifyMap(K_a, D_a, None, K_rgb, rgb_size, cv2.CV_32FC1)
# rectified-left -> rgb: X_rgb = R_la @ R1^T @ X_rect + t_la
R_ra = R_la @ R1.T
T_rgb_leftrect = np.eye(4)
T_rgb_leftrect[:3, :3], T_rgb_leftrect[:3, 3] = R_ra, t_la

print(f"Rectified: focal {f_rect:.2f} px, baseline {baseline_mm:.2f} mm, principal point ({cx_rect:.1f}, {cy_rect:.1f})"
      f" | RGB pinhole fx {K_rgb[0, 0]:.1f} at {rgb_size[0]}x{rgb_size[1]} | {len(names)} frames", flush=True)

json.dump({
    "rgb": {"width": rgb_size[0], "height": rgb_size[1], "fx": K_rgb[0, 0], "fy": K_rgb[1, 1],
            "cx": K_rgb[0, 2], "cy": K_rgb[1, 2], "model": "pinhole (undistorted)"},
    "left_rect": {"width": mono_size[0], "height": mono_size[1], "fx": f_rect, "fy": P1[1, 1],
                  "cx": cx_rect, "cy": cy_rect, "model": "pinhole (rectified)"},
    "baseline_mm": baseline_mm,
    "T_rgb_leftrect": T_rgb_leftrect.tolist(),
    "T_rgb_leftrect_note": "X_rgb = R @ X_leftrect + t, t in mm; depth PNGs are uint16 millimetres, 0 = invalid",
    "R_leftrect_left": R1.tolist(),
    "source_calibration": "calibration.json (device EEPROM at record time)",
    "extrinsic_refinement": refinement,
}, open(os.path.join(out, "cameras.json"), "w"), indent=1)


if args.geometry_only:
    json.dump([{"name": n, **frame_geo_info[i], "measured_deg": np.round(np.degrees(per_frame[i]), 4).tolist(),
                "smoothed_deg": np.round(np.degrees(per_frame_smooth[i]), 4).tolist()} for i, n in enumerate(names)],
              open(os.path.join(out, "geometry.json"), "w"), indent=1)
    raise SystemExit(0)


# ---- helpers ------------------------------------------------------------------------------------------------------
def rect_float(path, maps):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    return cv2.remap(img, maps[0], maps[1], cv2.INTER_LINEAR, borderValue=0)


def lr_difference(disp_lr, disp_rl):
    """|left disparity - right disparity at the matched pixel| (px); inf where the match falls outside the image."""
    h, w = disp_lr.shape
    x = np.arange(w, dtype=np.float32)[None, :] - disp_lr
    inside = (x >= 0) & (x <= w - 1)
    xi = np.clip(np.round(x), 0, w - 1).astype(np.int32)
    diff = np.abs(disp_lr - np.take_along_axis(disp_rl, xi, axis=1))
    diff[~inside] = np.inf
    return diff


def project_to_rgb(depth_mm):
    """Forward-splat rectified-left depth into the undistorted RGB image (2x2 footprint, nearest surface wins)."""
    v, u = np.nonzero(depth_mm)
    z = depth_mm[v, u].astype(np.float64)
    pts = np.stack([(u - cx_rect) / f_rect * z, (v - cy_rect) / P1[1, 1] * z, z], 1) @ R_ra.T + t_la
    zr = pts[:, 2]
    keep = zr > 1
    pts, zr = pts[keep], zr[keep]
    ur = K_rgb[0, 0] * pts[:, 0] / zr + K_rgb[0, 2] - 0.5  # footprint covers [floor(u), floor(u)+1]
    vr = K_rgb[1, 1] * pts[:, 1] / zr + K_rgb[1, 2] - 0.5
    u0, v0 = np.floor(ur).astype(np.int64), np.floor(vr).astype(np.int64)
    ui = np.concatenate([u0, u0 + 1, u0, u0 + 1])
    vi = np.concatenate([v0, v0, v0 + 1, v0 + 1])
    zz = np.tile(zr, 4)
    ok = (ui >= 0) & (ui < rgb_size[0]) & (vi >= 0) & (vi < rgb_size[1])
    ui, vi, zz = ui[ok], vi[ok], zz[ok]
    order = np.argsort(-zz, kind="stable")  # far first: with repeated indices the last (nearest) write wins
    out_d = np.zeros(rgb_size[::-1], np.float64)
    out_d[vi[order], ui[order]] = zz[order]
    return np.clip(np.round(out_d), 0, 65535).astype(np.uint16)


def write_ply(path, depth_mm, rgb):
    v, u = np.nonzero(depth_mm)
    z = depth_mm[v, u].astype(np.float32) / 1000.0
    xyz = np.stack([(u - K_rgb[0, 2]) / K_rgb[0, 0] * z, (v - K_rgb[1, 2]) / K_rgb[1, 1] * z, z], 1).astype(np.float32)
    col = rgb[v, u][:, ::-1].astype(np.uint8)
    rec_arr = np.empty(len(z), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec_arr["x"], rec_arr["y"], rec_arr["z"] = xyz.T
    rec_arr["red"], rec_arr["green"], rec_arr["blue"] = col.T
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(z)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        f.write(rec_arr.tobytes())


def row_error(left, right):
    """Median vertical offset (px) of SIFT matches between rectified images: 0 for a perfect calibration."""
    sift = cv2.SIFT_create(4000)
    k1, d1 = sift.detectAndCompute(left, None)
    k2, d2 = sift.detectAndCompute(right, None)
    if d1 is None or d2 is None:
        return None, 0
    m = [a for a, b in cv2.BFMatcher().knnMatch(d1, d2, k=2) if a.distance < 0.7 * b.distance]
    dy = np.array([k1[a.queryIdx].pt[1] - k2[a.trainIdx].pt[1] for a in m])
    dy = dy[np.abs(dy) < 5]
    return (float(np.median(np.abs(dy))) if dy.size else None), int(dy.size)


# ---- main loop ----------------------------------------------------------------------------------------------------
runner = FoundationStereoRunner(args.model, args.iters, args.scale)
video = cv2.VideoWriter(os.path.join(out, "preview.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                        (meta.get("measured_fps") or meta["fps"]) / args.stride, (640 * 3, 480))
frame_info, times, row_errors = [], [], []
by_index = {f"{f['index']:06d}": f for f in frames_meta}
for i, name in enumerate(names):
    t0 = time.perf_counter()
    done = args.resume and all(os.path.exists(os.path.join(out, s, name + ".png"))
                               for s in ("rgb", "depth", "depth_left"))
    if done:  # finished by an earlier, interrupted run: reuse it for the preview video and the statistics
        rgb = cv2.imread(os.path.join(out, "rgb", name + ".png"), cv2.IMREAD_COLOR)
        depth_rgb = cv2.imread(os.path.join(out, "depth", name + ".png"), cv2.IMREAD_UNCHANGED)
        depth_left = cv2.imread(os.path.join(out, "depth_left", name + ".png"), cv2.IMREAD_UNCHANGED)
        t_fs = 0.0
    else:
        left = rect_float(os.path.join(rec, "left", name + ".png"), map_l)
        R2k = right_rect_rotation(per_frame_smooth[i])
        map_r = cv2.initUndistortRectifyMap(K_r, D_r, R2k, P2, mono_size, cv2.CV_32FC1)
        right = rect_float(os.path.join(rec, "right", name + ".png"), map_r)
        disp = runner.disparity(left, right)
        ok = valid_l & (disp > fb / (args.max_m * 1000.0)) & (disp < fb / (args.min_m * 1000.0))
        # raw disparity (1/64 px) and left-right disagreement (1/100 px, 65535 = no match) are kept so the validity
        # threshold can be changed later without re-running FoundationStereo
        cv2.imwrite(os.path.join(out, "disp_left", name + ".png"),
                    np.clip(np.round(disp * 64), 0, 65535).astype(np.uint16))
        if not args.no_lr:
            disp_rl = runner.disparity(np.ascontiguousarray(right[:, ::-1]), np.ascontiguousarray(left[:, ::-1]))[:, ::-1]
            diff = lr_difference(disp, np.ascontiguousarray(disp_rl))
            ok &= diff < args.lr_threshold
            cv2.imwrite(os.path.join(out, "lr_diff", name + ".png"),
                        np.where(np.isfinite(diff), np.clip(np.round(diff * 100), 0, 65534), 65535).astype(np.uint16))
        depth_left = np.zeros(disp.shape, np.uint16)
        depth_left[ok] = np.clip(np.round(fb / disp[ok]), 0, 65535).astype(np.uint16)
        t_fs = time.perf_counter() - t0

        rgb_raw = cv2.imread(os.path.join(rec, "rgb", name + ".jpg"), cv2.IMREAD_COLOR)
        rgb = cv2.remap(rgb_raw, map_a[0], map_a[1], cv2.INTER_LINEAR)
        depth_rgb = project_to_rgb(depth_left)
        left8 = np.clip(left + 0.5, 0, 255).astype(np.uint8)

        cv2.imwrite(os.path.join(out, "rgb", name + ".png"), rgb, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        cv2.imwrite(os.path.join(out, "depth", name + ".png"), depth_rgb)
        cv2.imwrite(os.path.join(out, "left_rect", name + ".png"), left8)
        cv2.imwrite(os.path.join(out, "depth_left", name + ".png"), depth_left)
        if i == 0 or (args.ply_every and i % args.ply_every == 0):
            write_ply(os.path.join(out, "clouds", name + ".ply"), depth_rgb, rgb)
        if i == 0 or i % 10 == 0:
            err, n = row_error(left8, np.clip(right + 0.5, 0, 255).astype(np.uint8))
            if err is not None:
                row_errors.append(err)

    small_rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_AREA)
    small_d = cv2.resize(depth_rgb, (640, 480), interpolation=cv2.INTER_NEAREST)
    col = colorize(small_d, near=args.min_m * 1000, far=min(args.max_m, 6.0) * 1000)
    over = small_rgb.copy()
    m = small_d > 0
    over[m] = (0.45 * small_rgb[m] + 0.55 * col[m]).astype(np.uint8)
    for tile, txt in ((small_rgb, f"rgb {name}"), (col, f"FS {runner.name} fill {100 * (depth_rgb > 0).mean():.0f}%"),
                      (over, "overlay")):
        for c, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            cv2.putText(tile, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, th)
    video.write(np.hstack([small_rgb, col, over]))

    if not done:
        times.append(time.perf_counter() - t0)
    info = dict(by_index.get(name, {"index": int(name)}))
    info["right_correction_deg"] = dict(zip(("pitch", "yaw", "roll"), np.round(np.degrees(per_frame_smooth[i]), 4).tolist()))
    info.update(frame_geo_info[i])
    info.update({"name": name, "fill_rgb_pct": round(100 * float((depth_rgb > 0).mean()), 1),
                 "fill_left_pct": round(100 * float((depth_left > 0).mean()), 1)})
    frame_info.append(info)
    if done:
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(names)}  reused from the earlier run", flush=True)
    elif not times:
        pass
    elif len(times) == 1 or (i + 1) % 5 == 0 or i + 1 == len(names):
        left_s = np.mean(times[-5:]) * (len(names) - i - 1)
        print(f"  {i + 1}/{len(names)}  FS {t_fs:.1f} s, total {times[-1]:.1f} s/frame, fill rgb "
              f"{info['fill_rgb_pct']:.0f}% left {info['fill_left_pct']:.0f}%, ~{left_s / 60:.1f} min left", flush=True)
video.release()

json.dump(frame_info, open(os.path.join(out, "frames.json"), "w"), indent=1)
summary = {
    "recording": rec, "model": args.model, "iters": args.iters, "scale": args.scale, "lr_check": not args.no_lr,
    "frames": len(names), "s_per_frame": round(float(np.mean(times[1:] or times)), 2) if times else None,
    "rectified_focal_px": round(f_rect, 3), "baseline_mm": round(baseline_mm, 3),
    "rectification_row_error_px_median": round(float(np.median(row_errors)), 3) if row_errors else None,
    "extrinsic_refinement": refinement,
    "fill_rgb_pct_mean": round(float(np.mean([f["fill_rgb_pct"] for f in frame_info])), 1),
    "fill_left_pct_mean": round(float(np.mean([f["fill_left_pct"] for f in frame_info])), 1),
    "depth_precision_mm_at_1m_for_0.25px": round(1000.0 ** 2 * 0.25 / fb, 2),
}
json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
print("SUMMARY " + json.dumps(summary), flush=True)
