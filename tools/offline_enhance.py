#!/usr/bin/env python3
"""Offline enhancement of the frozen ship dataset with independent restorers.

Each method runs once over train/val/test with its official pretrained
weights (frozen, no fine-tuning) at the original 640x640 resolution. The
output is a complete YOLO dataset (images + copied labels + data.yaml) on
which stock YOLO11n is then trained with the shared protocol.

Methods
-------
adair   AdaIR (ICLR 2025), https://github.com/c-yn/AdaIR, checkpoint
        ``adair5d.ckpt`` (5-task all-in-one, includes dehazing and
        low-light). Runs in-process with batching; optional FP16, where the
        FFT inside AdaIR is forced to FP32 because cuFFT half precision only
        supports power-of-two sizes.
glade   GLADE-Net (Ecological Informatics 2025),
        https://github.com/riotu-lab/GLADE-Net. Phase 1 (Uformer) via the
        official ``tools/inference_grid.py`` with ``--tile 640`` (one grid per
        640x640 image), then Phase 2 (Laplacian-guided GAN) via the official
        ``Phase_2/code/test.py``. Weights: SateHaze1k (haze).
ipc     IPC-Dehaze (CVPR 2025), https://github.com/Jiayi-Fu/IPC-Dehaze, via
        the official ``inference.py`` (FP32, 8 iterations, unchanged).

Example (Colab)
---------------
python tools/offline_enhance.py --method adair \
    --repo /content/AdaIR --weights /content/AdaIR/ckpt/adair5d.ckpt \
    --src /content/drive/MyDrive/ship_detection/data \
    --dst /content/enhanced/adair --batch 4 --fp16

Use ``--limit 20`` first to check outputs and timing on a few images.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def list_images(folder: Path) -> list[Path]:
    images = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    stems = [p.stem for p in images]
    if len(stems) != len(set(stems)):
        raise ValueError(f"Duplicate image stems in {folder}; outputs are named <stem>.png.")
    return images


def stage_inputs(images: list[Path], staging: Path) -> Path:
    """Folder that contains only the images to process (official scripts glob
    everything in their input folder, e.g. Ultralytics .npy caches)."""
    staging.mkdir(parents=True, exist_ok=True)
    for image in images:
        link = staging / image.name
        if not link.exists():
            os.symlink(image.resolve(), link)
    return staging


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd), f"(cwd={cwd})", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def verify_outputs(images: list[Path], out_dir: Path) -> None:
    from PIL import Image

    missing, wrong = [], []
    for image in images:
        out = out_dir / f"{image.stem}.png"
        if not out.is_file():
            missing.append(out.name)
            continue
        with Image.open(image) as a, Image.open(out) as b:
            if a.size != b.size:
                wrong.append(f"{image.name}: {a.size} -> {b.size}")
    if missing or wrong:
        raise RuntimeError(
            f"Output check failed in {out_dir}: {len(missing)} missing, {len(wrong)} size changes. "
            f"First: {(missing + wrong)[:5]}"
        )


def gpu_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - informational only
        return "unknown"


# ---------------------------------------------------------------------------
# AdaIR (in-process)
# ---------------------------------------------------------------------------
class _FFTFloat32:
    """Context manager: run torch.fft.fft2/ifft2 in FP32 under autocast."""

    def __enter__(self):
        import torch

        self._fft2, self._ifft2 = torch.fft.fft2, torch.fft.ifft2

        def fft2(x, *args, **kwargs):
            if x.dtype in (torch.float16, torch.bfloat16):
                x = x.float()
            return self._fft2(x, *args, **kwargs)

        def ifft2(x, *args, **kwargs):
            if x.dtype == torch.complex32:
                x = x.to(torch.complex64)
            return self._ifft2(x, *args, **kwargs)

        torch.fft.fft2, torch.fft.ifft2 = fft2, ifft2
        return self

    def __exit__(self, *exc):
        import torch

        torch.fft.fft2, torch.fft.ifft2 = self._fft2, self._ifft2
        return False


def load_adair(repo: Path, weights: Path, device):
    import torch

    sys.path.insert(0, str(repo))
    from net.model import AdaIR  # noqa: E402  (official repository)

    net = AdaIR(decoder=True)
    checkpoint = torch.load(str(weights), map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    state = {k[len("net."):]: v for k, v in state.items() if k.startswith("net.")} or state
    net.load_state_dict(state, strict=True)
    return net.to(device).eval()


def enhance_adair(images, out_dir: Path, args) -> None:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_adair(Path(args.repo), Path(args.weights), device)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = 16  # AdaIR's test loader crops to multiples of 16; we pad instead
    for start in range(0, len(images), args.batch):
        chunk = images[start:start + args.batch]
        arrays = [np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0 for p in chunk]
        sizes = {a.shape for a in arrays}
        if len(sizes) != 1:
            raise ValueError("Batch images must share one size; use --batch 1.")
        h, w = arrays[0].shape[:2]
        batch = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(device)
        ph, pw = (-h) % base, (-w) % base
        if ph or pw:
            batch = F.pad(batch, (0, pw, 0, ph), mode="reflect")
        with torch.no_grad(), _FFTFloat32(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=args.fp16 and device.type == "cuda"
        ):
            restored = net(batch)
        restored = restored[..., :h, :w].float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        for path, img in zip(chunk, restored):
            Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8)).save(out_dir / f"{path.stem}.png")
        print(f"  AdaIR {start + len(chunk)}/{len(images)}", flush=True)


# ---------------------------------------------------------------------------
# GLADE-Net (official scripts, two phases)
# ---------------------------------------------------------------------------
# The released Phase-2 weights were trained with the first-level Laplacian
# extractor (5x5 Gaussian buffer ``get_g_nopadding.kernel``; paper Sec. 3.2:
# "we utilize only the first level of the Laplacian pyramid"), which is
# commented out in architecture.py. The active default is a 3x3 directional
# gradient, and ``load_network`` silently drops checkpoint keys the model does
# not have. We therefore swap in the level-one class (verbatim from the
# commented block) on a copy of the code and require a strict load.
_GLADE_LEVEL_ONE = """class Get_gradient_nopadding(nn.Module):
    # GLADE-Net first-level Laplacian (verbatim from the commented 'level one'
    # block of the official architecture.py), selected to match the weights.
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        self.register_buffer('kernel', kernel)

    def conv_gauss(self, img, kernel):
        n_channels, _, kw, kh = kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, kernel, groups=n_channels)

    def forward(self, img):
        kernel = self.kernel.to(device=img.device, dtype=img.dtype)
        filtered = self.conv_gauss(img, kernel)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self.conv_gauss(new_filter, kernel)
        diff = img - filtered
        return diff


"""

_GLADE_CHECK = r"""
import sys, torch
sys.path.insert(0, '.')
from models.modules.architecture import SPSRNet
net = SPSRNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, upscale=1, norm_type=None,
              act_type='leakyrelu', mode='CNA', upsample_mode='upconv')
state = torch.load(sys.argv[1], map_location='cpu')
missing, unexpected = [], []
model_state = net.state_dict()
for key in state:
    if key not in model_state:
        unexpected.append(key)
    elif tuple(state[key].shape) != tuple(model_state[key].shape):
        unexpected.append(f"{key}: {tuple(state[key].shape)} vs {tuple(model_state[key].shape)}")
missing = [key for key in model_state if key not in state]
assert not missing and not unexpected, f"missing={missing[:10]} unexpected={unexpected[:10]}"
net.load_state_dict(state, strict=True)
k = torch.tensor([[.05, .25, .4, .25, .05]])
expected = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
got = state['get_g_nopadding.kernel'].float()
assert torch.allclose(got, expected, atol=1e-6), f"checkpoint kernel differs: {got[0, 0].tolist()}"
print('GLADE-Net Phase-2 weights: strict load OK, 5x5 first-level Laplacian kernel verified')
"""


def prepare_glade_phase2(repo: Path, weights2: Path, work: Path) -> Path:
    code = work / "phase2_code"
    shutil.copytree(repo / "Phase_2/code", code)
    arch = code / "models/modules/architecture.py"
    text = arch.read_text()
    start = text.find("class Get_gradient_nopadding(nn.Module):")
    end = text.find("################################", start)
    if start < 0 or end < 0 or "kernel_v = [[0, -1, 0]" not in text[start:end]:
        raise RuntimeError("GLADE-Net architecture.py layout changed; check Get_gradient_nopadding manually.")
    arch.write_text(text[:start] + _GLADE_LEVEL_ONE + text[end:])
    check = work / "glade_phase2_check.py"
    check.write_text(_GLADE_CHECK)
    run([sys.executable, str(check), str(weights2)], cwd=code)
    return code


def enhance_glade(images, out_dir: Path, args, work: Path) -> None:
    repo = Path(args.repo).resolve()
    staged = stage_inputs(images, work / "input")
    phase1 = work / "phase1"
    cmd = [sys.executable, str(repo / "tools/inference_grid.py"), "--weights", str(Path(args.weights).resolve()),
           "--input", str(staged), "--output", str(phase1), "--tile", "640"]
    if args.fp16:
        cmd.append("--fp16")
    run(cmd, cwd=repo)
    phase2_code = prepare_glade_phase2(repo, Path(args.weights2).resolve(), work)
    phase2_root = work / "phase2"
    opt = {
        "name": "glade_phase2",
        "model": "spsr",
        "scale": 1,
        "gpu_ids": [0],
        "datasets": {"test_1": {"name": "out", "mode": "LR", "dataroot_LR": str(phase1)}},
        "path": {"root": str(phase2_root), "pretrain_model_G": str(Path(args.weights2).resolve())},
        "network_G": {"which_model_G": "spsr_net", "norm_type": None, "mode": "CNA", "nf": 64, "nb": 23,
                      "in_nc": 3, "out_nc": 3, "gc": 32, "group": 1},
    }
    opt_path = work / "phase2_test.json"
    opt_path.write_text(json.dumps(opt, indent=2))
    run([sys.executable, "test.py", "-opt", str(opt_path)], cwd=phase2_code)
    result_dir = phase2_root / "results" / "glade_phase2" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    for image in images:
        shutil.copy2(result_dir / f"{image.stem}.png", out_dir / f"{image.stem}.png")


# ---------------------------------------------------------------------------
# IPC-Dehaze (official script)
# ---------------------------------------------------------------------------
def enhance_ipc(images, out_dir: Path, args, work: Path) -> None:
    repo = Path(args.repo).resolve()
    staged = stage_inputs(images, work / "input")
    out_dir.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "inference.py", "-i", str(staged), "-o", str(out_dir.resolve()),
         "--predictor_path", str(Path(args.weights).resolve()),
         "--critic_path", str(Path(args.weights2).resolve()), "-n", "8"], cwd=repo)


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, choices=("adair", "glade", "ipc"))
    parser.add_argument("--repo", required=True, help="clone of the official repository")
    parser.add_argument("--weights", required=True, help="adair: .ckpt | glade: Phase-1 .ckpt | ipc: predictor.pth")
    parser.add_argument("--weights2", help="glade: Phase-2 .pth | ipc: critic.pth")
    parser.add_argument("--src", required=True, help="dataset root with {train,val,test}/{images,labels}")
    parser.add_argument("--dst", required=True, help="output dataset root (must not exist)")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--batch", type=int, default=1, help="adair only")
    parser.add_argument("--fp16", action="store_true", help="adair and glade Phase 1")
    parser.add_argument("--limit", type=int, default=0, help="process only the first N images per split (dry run)")
    args = parser.parse_args()

    if args.method in ("glade", "ipc") and not args.weights2:
        parser.error(f"--weights2 is required for {args.method}")
    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        raise FileExistsError(f"{dst} exists; choose a new --dst (nothing is overwritten).")

    timing = {"method": args.method, "fp16": args.fp16, "batch": args.batch, "limit": args.limit,
              "gpu": gpu_name(), "python": platform.python_version(), "splits": {}}
    for split in args.splits:
        images = list_images(src / split / "images")
        if args.limit:
            images = images[: args.limit]
        out_dir = dst / split / "images"
        print(f"[{args.method}] {split}: {len(images)} images", flush=True)
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix=f"enh_{args.method}_{split}_") as tmp:
            if args.method == "adair":
                enhance_adair(images, out_dir, args)
            elif args.method == "glade":
                enhance_glade(images, out_dir, args, Path(tmp))
            else:
                enhance_ipc(images, out_dir, args, Path(tmp))
        seconds = time.perf_counter() - t0
        verify_outputs(images, out_dir)
        label_src, label_dst = src / split / "labels", dst / split / "labels"
        label_dst.mkdir(parents=True, exist_ok=True)
        for image in images:
            label = label_src / f"{image.stem}.txt"
            if label.is_file():
                shutil.copy2(label, label_dst / label.name)
        timing["splits"][split] = {"images": len(images), "seconds": round(seconds, 2),
                                   "seconds_per_image_wall": round(seconds / max(len(images), 1), 4)}
        print(f"[{args.method}] {split}: {seconds:.1f}s ({seconds / max(len(images), 1):.3f}s/img, wall clock)")

    (dst / "data.yaml").write_text(
        f"path: {dst.resolve()}\ntrain: train/images\nval: val/images\ntest: test/images\nnc: 1\nnames:\n  0: ship\n"
    )
    (dst / "enhancement_timing.json").write_text(json.dumps(timing, indent=2))
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
