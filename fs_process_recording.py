"""Run FoundationStereo on a recording made with depth_live.py --record-seconds (runs in WSL on the GPU).

Writes into the recording folder:
    fs_depth_<model>/000000.png   FoundationStereo depth, uint16 mm (0 = invalid), aligned to left/000000.png
    compare_<model>.avi           rectified left | device stereo depth | FoundationStereo depth (MJPEG)
    fs_summary_<model>.json       timing and agreement with the device depth

    python fs_process_recording.py                     # newest recording, ViT-small, every frame
    python fs_process_recording.py --model vitl --iters 32 --stride 2 REC_DIR
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np

from fs_model import FoundationStereoRunner, colorize, disparity_to_mm, label

parser = argparse.ArgumentParser()
parser.add_argument("recording", nargs="?", default=None, help="Recording folder (default: newest)")
parser.add_argument("--model", default="vits", choices=["vits", "vitl"])
parser.add_argument("--iters", type=int, default=16, help="Refinement iterations (repo default 32)")
parser.add_argument("--scale", type=float, default=1.0, help="Inference resolution scale")
parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
parser.add_argument("--max-frames", type=int, default=0, help="0 = all")
args = parser.parse_args()

rec = args.recording or max(glob.glob(os.path.expanduser("~/oakffc3p/recordings/*/meta.json")), key=os.path.getmtime)
rec = os.path.dirname(rec) if rec.endswith("meta.json") else rec
meta = json.load(open(os.path.join(rec, "meta.json")))
frames = sorted(glob.glob(os.path.join(rec, "left", "*.png")))[::args.stride]
if args.max_frames:
    frames = frames[:args.max_frames]
fb = meta["rectified_focal_px_from_calib"] * meta["baseline_mm"]
print(f"{rec}: {len(frames)} frames to process, focal {meta['rectified_focal_px_from_calib']:.1f} px x "
      f"baseline {meta['baseline_mm']:.2f} mm", flush=True)

runner = FoundationStereoRunner(args.model, args.iters, args.scale)
out_dir = os.path.join(rec, f"fs_depth_{args.model}")
os.makedirs(out_dir, exist_ok=True)
w, h = meta["width"], meta["height"]
video_fps = meta.get("measured_fps") or meta["fps"]
video = cv2.VideoWriter(os.path.join(rec, f"compare_{args.model}.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                        video_fps / args.stride, (w * 3, h))

times, ratios, fills = [], [], []
frame_info = {f"{f['index']:06d}.png": f for f in meta.get("frames", [])}
blur_limit = meta.get("blur_flag_px", 2.0)
blurred = 0
for i, left_path in enumerate(frames):
    name = os.path.basename(left_path)
    left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(os.path.join(rec, "right", name), cv2.IMREAD_GRAYSCALE)
    dev = cv2.imread(os.path.join(rec, "depth", name), cv2.IMREAD_UNCHANGED)
    if dev.shape != left.shape:  # device depth is recorded at half resolution
        dev = cv2.resize(dev, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_NEAREST)
    t0 = time.perf_counter()
    disp = runner.disparity(left, right)
    times.append(time.perf_counter() - t0)
    fs = disparity_to_mm(disp, fb)
    fs[left == 0] = 0  # no image there (rectification border)
    cv2.imwrite(os.path.join(out_dir, name), fs)

    both = (dev > 0) & (fs > 0)
    if both.sum() > 1000:
        ratios.append(float(np.median(fs[both].astype(np.float64) / dev[both])))
    fills.append((float((dev > 0).mean()), float((fs > 0).mean())))
    imu = (frame_info.get(name) or {}).get("imu")
    left_tile = label(cv2.cvtColor(left, cv2.COLOR_GRAY2BGR), f"left  frame {name[:-4]}")
    if imu:
        is_blurry = imu["blur_px"] > blur_limit
        blurred += is_blurry
        txt = f"{imu['gyro_dps']:.0f} deg/s  blur {imu['blur_px']:.1f} px" + ("  MOTION BLUR" if is_blurry else "")
        for c, th in (((0, 0, 0), 4), ((0, 0, 255) if is_blurry else (0, 255, 0), 1)):
            cv2.putText(left_tile, txt, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, th)
    tiles = [left_tile,
             label(colorize(dev), f"device stereo  fill {100 * fills[-1][0]:.0f}%"),
             label(colorize(fs), f"FoundationStereo {runner.name}  {1000 * times[-1]:.0f} ms")]
    video.write(np.hstack(tiles))
    if i == 0 or (i + 1) % 10 == 0 or i + 1 == len(frames):
        print(f"  {i + 1}/{len(frames)}  {1000 * np.mean(times[-10:]):.0f} ms/frame"
              f"  FS/device depth ratio {np.median(ratios[-10:]) if ratios else float('nan'):.3f}", flush=True)
video.release()

summary = {
    "recording": rec, "model": args.model, "iters": args.iters, "scale": args.scale, "frames": len(frames),
    "ms_per_frame_mean": round(1000 * float(np.mean(times[1:] or times)), 1),
    "fs_over_device_depth_median": round(float(np.median(ratios)), 4) if ratios else None,
    "device_fill_mean_pct": round(100 * float(np.mean([f[0] for f in fills])), 1),
    "fs_fill_mean_pct": round(100 * float(np.mean([f[1] for f in fills])), 1),
    "frames_flagged_motion_blur": blurred,
}
json.dump(summary, open(os.path.join(rec, f"fs_summary_{args.model}.json"), "w"), indent=1)
print("SUMMARY " + json.dumps(summary), flush=True)
