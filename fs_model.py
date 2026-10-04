"""FoundationStereo (NVlabs, as used by projectaria_gen2_depth_from_stereo) loading and inference helpers."""
import os
import sys
import types

# Without this the caching allocator fragments to ~7.8 GB reserved on the 8 GB RTX 5050 and Windows starts paging GPU
# memory to system RAM: every call after the first gets ~5x slower (measured 15 s -> 69 s per 1280x800 ViT-L pass)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
FS_PATH = os.path.join(ROOT, "projectaria_gen2_depth_from_stereo", "FoundationStereo")
CKPTS = {"vits": os.path.expanduser("~/oakffc3p/fs_ckpts/11-33-40/model_best_bp2.pth"),
         "vitl": os.path.expanduser("~/oakffc3p/fs_ckpts/23-51-11/model_best_bp2.pth")}


class FoundationStereoRunner:
    def __init__(self, model="vits", iters=8, scale=1.0):
        sys.path.insert(0, FS_PATH)
        if "open3d" not in sys.modules:  # only used by FoundationStereo's visualization helpers
            try:
                import open3d  # noqa: F401
            except ImportError:
                sys.modules["open3d"] = types.ModuleType("open3d")
        from core.foundation_stereo import FoundationStereo
        from core.utils.utils import InputPadder
        from omegaconf import OmegaConf

        ckpt_path = CKPTS[model]
        cfg = OmegaConf.load(os.path.join(os.path.dirname(ckpt_path), "cfg.yaml"))
        if "vit_size" not in cfg:
            cfg["vit_size"] = "vitl"
        net = FoundationStereo(cfg)
        net.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"])
        self.net = net.cuda().eval()
        self.padder_cls, self.iters, self.scale, self.name = InputPadder, iters, scale, f"{cfg.vit_size} it{iters}"
        print(f"FoundationStereo {cfg.vit_size} ({model}) loaded on {torch.cuda.get_device_name(0)}, "
              f"{iters} iterations, scale {scale}", flush=True)

    @torch.no_grad()
    def disparity(self, left, right):
        """Disparity in pixels at the input resolution, from rectified uint8 grayscale (or BGR) images."""
        h, w = left.shape[:2]
        if self.scale != 1.0:
            size = (max(32, int(w * self.scale)), max(32, int(h * self.scale)))
            left = cv2.resize(left, size, interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, size, interpolation=cv2.INTER_AREA)
        t = []
        for img in (left, right):
            x = torch.from_numpy(np.ascontiguousarray(img)).cuda().float()
            x = x[None, None].expand(1, 3, -1, -1) if x.ndim == 2 else x.permute(2, 0, 1)[None]
            t.append(x)
        padder = self.padder_cls(t[0].shape, divis_by=32, force_square=False)
        lp, rp = padder.pad(*t)
        with torch.amp.autocast("cuda"):
            disp = self.net.forward(lp, rp, iters=self.iters, test_mode=True)
        disp = padder.unpad(disp.float()).squeeze().cpu().numpy()
        del lp, rp, t
        torch.cuda.empty_cache()  # keep reserved memory low between calls (see PYTORCH_CUDA_ALLOC_CONF above)
        if disp.shape != (h, w):  # back to the input resolution, rescaling disparity values
            disp = cv2.resize(disp, (w, h), interpolation=cv2.INTER_LINEAR) * (w / disp.shape[1])
        return disp


def disparity_to_mm(disp, focal_baseline_mm, min_disp=0.5):
    depth = np.zeros(disp.shape, np.float32)
    ok = disp > min_disp
    depth[ok] = focal_baseline_mm / disp[ok]
    return np.clip(depth, 0, 65535).astype(np.uint16)


def colorize(depth_mm, near=250.0, far=5000.0):
    """TURBO on log distance (near = red, far = blue, 0 = black), same scale as view_depth.py."""
    level = 1.0 - np.clip(np.log(np.maximum(depth_mm, 1) / near) / np.log(far / near), 0, 1)
    color = cv2.applyColorMap((level * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    color[depth_mm == 0] = 0
    return color


def label(img, text):
    for c, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, th)
    return img
