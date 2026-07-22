# Colab cells for FaPN-Prefusion formal training

Run one cell at a time. Cell 5 is the only cell that starts formal training.
Set `VARIANT` to `"baseline"` or `"inceptiondw"` and run the two experiments
separately. No dataset or checkpoint is trained from Google Drive directly.

## Cell 1 — Drive and private clone

The token is hidden by `getpass`, passed only through an ephemeral
`GIT_ASKPASS` environment variable, never placed in the URL/remote/log, and
removed immediately after clone.

```python
from google.colab import drive
drive.mount("/content/drive")

import getpass, os, stat, subprocess, textwrap
from pathlib import Path

REPO_URL = "https://github.com/HoverdZ/ship-yolo.git"
BRANCH = "feature/fapn-prefusion"
REPO = Path("/content/ship-yolo")
if REPO.exists():
    raise FileExistsError(f"Refusing to overwrite {REPO}")

token = getpass.getpass("GitHub token (hidden; never paste it into notebook text): ")
askpass = Path("/content/.ship_yolo_askpass.py")
askpass.write_text(textwrap.dedent("""
    #!/usr/bin/env python3
    import os, sys
    print("x-access-token" if "Username" in sys.argv[1] else os.environ["SHIP_GITHUB_TOKEN"])
""").lstrip(), encoding="utf-8")
askpass.chmod(askpass.stat().st_mode | stat.S_IXUSR)
env = os.environ.copy()
env.update({
    "GIT_ASKPASS": str(askpass),
    "GIT_TERMINAL_PROMPT": "0",
    "SHIP_GITHUB_TOKEN": token,
})
try:
    subprocess.run(
        [
            "git", "-c", "credential.helper=", "clone", "--branch", BRANCH,
            "--single-branch", REPO_URL, str(REPO),
        ],
        check=True,
        env=env,
    )
finally:
    token = None
    env.pop("SHIP_GITHUB_TOKEN", None)
    os.environ.pop("SHIP_GITHUB_TOKEN", None)
    askpass.unlink(missing_ok=True)

# Pin the exact branch head resolved by this clone and record it with the run.
FINAL_COMMIT = subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
).strip()
subprocess.run(["git", "-C", str(REPO), "checkout", "--detach", FINAL_COMMIT], check=True)
print("Checked out exact commit:", FINAL_COMMIT)
print(subprocess.check_output(["git", "-C", str(REPO), "remote", "-v"], text=True))
```

## Cell 2 — pinned runtime and DCNv2 check

This installs only Ultralytics. It does not replace PyTorch or Torchvision.

```python
import subprocess, sys
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.92"],
    check=True,
)

import torch, torchvision, ultralytics
from torchvision.ops import DeformConv2d

assert ultralytics.__version__ == "8.4.92", ultralytics.__version__
assert torch.__version__ == "2.11.0+cu128", torch.__version__
assert torchvision.__version__ == "0.26.0+cu128", torchvision.__version__
assert torch.cuda.is_available()
assert "L4" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)

op = DeformConv2d(8, 8, 3, padding=1, groups=8, bias=True).cuda()
x = torch.randn(1, 8, 8, 8, device="cuda")
offset = torch.zeros(1, 144, 8, 8, device="cuda")
mask = torch.full((1, 72, 8, 8), 0.5, device="cuda")
y = op(x, offset, mask)
assert torch.isfinite(y).all()

sys.path.insert(0, str(REPO))
from tools.probe_fapn_prefusion_amp import run_probe
amp_probe = run_probe("cuda:0", amp=True)
assert amp_probe["all_checks_passed"], amp_probe
print(amp_probe)
```

## Cell 3 — multithread copy to local disk and data audit

Adjust `DRIVE_DATA_ROOT` only. Expected layout is
`images/{train,val,test}` and `labels/{train,val,test}`. `copy2` preserves
zero-byte label files.

```python
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil, yaml

DRIVE_DATA_ROOT = Path("/content/drive/MyDrive/ship_detection/dataset")
LOCAL_DATA_ROOT = Path("/content/datasets/ship_detection")
LOCAL_DATA_ROOT.mkdir(parents=True, exist_ok=True)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def copy_tree_files(source: Path, destination: Path, workers: int = 24):
    if not source.is_dir():
        raise FileNotFoundError(source)
    files = [path for path in source.rglob("*") if path.is_file()]
    def copy_one(path: Path):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)  # includes empty label files
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(copy_one, files))

for split in ("train", "val", "test"):
    copy_tree_files(DRIVE_DATA_ROOT / "images" / split, LOCAL_DATA_ROOT / "images" / split)
    copy_tree_files(DRIVE_DATA_ROOT / "labels" / split, LOCAL_DATA_ROOT / "labels" / split)

expected_images = {"train": 2582, "val": 842, "test": 874}
audit = {}
for split, expected in expected_images.items():
    images = [p for p in (LOCAL_DATA_ROOT / "images" / split).rglob("*")
              if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    labels = list((LOCAL_DATA_ROOT / "labels" / split).rglob("*.txt"))
    audit[split] = {"images": len(images), "labels": len(labels),
                    "empty_labels": sum(p.stat().st_size == 0 for p in labels)}
    assert len(images) == expected, (split, len(images), expected)

LOCAL_DATA_YAML = LOCAL_DATA_ROOT / "data.yaml"
LOCAL_DATA_YAML.write_text(yaml.safe_dump({
    "path": str(LOCAL_DATA_ROOT),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": 1,
    "names": ["ship"],
}, sort_keys=False), encoding="utf-8")
print(audit)
print(LOCAL_DATA_YAML.read_text())
```

## Cell 4 — build real init.pt files and verify manifests

This cell does not train. Both targets start only from official `yolo11n.pt`.
Generated checkpoints and matching manifests are stored in a fixed Drive
initialization directory so a runtime restart does not lose them.

```python
import json, shutil
from pathlib import Path
from ultralytics import YOLO

%cd /content/ship-yolo
OFFICIAL_WEIGHTS = Path("/content/yolo11n.pt")
if not OFFICIAL_WEIGHTS.is_file():
    downloaded = Path(YOLO("yolo11n.pt").ckpt_path)
    shutil.copy2(downloaded, OFFICIAL_WEIGHTS)

DRIVE_INIT_DIR = Path("/content/drive/MyDrive/ship_detection/fapn_prefusion_init")
DRIVE_INIT_DIR.mkdir(parents=True, exist_ok=True)

from tools.fapn_prefusion_utils import prepare_initialization, validate_init_manifest
from tools.fapn_prefusion_profile import profile_variant
from tools.fapn_prefusion_utils import variant_config, write_json

prepared = {}
for variant in ("baseline", "inceptiondw"):
    result = prepare_initialization(
        variant,
        weights=OFFICIAL_WEIGHTS,
        output_dir=DRIVE_INIT_DIR,
        seed=0,
    )
    profile_payload = profile_variant(variant, imgsz=640)
    profile_path = DRIVE_INIT_DIR / Path(variant_config(variant)["profile"]).name
    write_json(profile_path, profile_payload)
    check = validate_init_manifest(result["init_pt"], result["manifest"])
    assert check["all_checks_passed"], check
    prepared[variant] = {**result, "profile": str(profile_path)}
    transfer = result["weight_transfer"]
    print(variant, {
        "init_pt": result["init_pt"],
        "inherited_state_tensors": transfer["inherited_state_tensors"],
        "total_state_tensors": transfer["total_state_tensors"],
        "inherited_parameter_elements": transfer["inherited_parameter_elements"],
        "target_parameter_elements": transfer["target_parameter_elements"],
        "ratio": transfer["parameter_element_inheritance_ratio"],
        "manifest_passed": check["all_checks_passed"],
    })
```

## Cell 5 — official Python API formal training

Choose one variant. If a run contains `last.pt`, this refuses to overwrite and
points to Cell 6. A failed directory without `last.pt` is moved to a timestamped
`crashed_backup` directory. Official tqdm/logger/model-info formatting remains
active.

```python
import warnings
from functools import partial
from pathlib import Path

VARIANT = "baseline"  # change to "inceptiondw" for the second formal run

from tools.fapn_prefusion_utils import (
    install_safe_prefusion_flops,
    prepare_formal_run_directory,
    register_modules,
    variant_config,
    verify_prefusion_trainer_initialization,
)
config = variant_config(VARIANT)
INIT_PT = DRIVE_INIT_DIR / Path(config["init_pt"]).name
MANIFEST = DRIVE_INIT_DIR / Path(config["manifest"]).name
PROFILE_JSON = DRIVE_INIT_DIR / Path(config["profile"]).name
PROJECT = Path("/content/drive/MyDrive/ship_detection/runs")
RUN_NAME = config["experiment_name"]

run_dir, crashed_backup = prepare_formal_run_directory(PROJECT, RUN_NAME)
print("run_dir:", run_dir, "crashed_backup:", crashed_backup)

register_modules()
restore_flops = install_safe_prefusion_flops(PROFILE_JSON)
try:
    from ultralytics import YOLO

    model = YOLO(str(INIT_PT))
    verifier_callback = partial(
        verify_prefusion_trainer_initialization,
        manifest_path=MANIFEST,
    )
    model.add_callback(
        "on_pretrain_routine_end",
        verifier_callback,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*compute_grad_input does not have a deterministic implementation.*",
            category=UserWarning,
        )
        results = model.train(
            data=str(LOCAL_DATA_YAML),
            epochs=150,
            imgsz=640,
            batch=8,
            workers=2,
            device=0,
            seed=0,
            deterministic=True,
            optimizer="auto",
            amp=True,
            val=True,
            plots=True,
            patience=100,
            project=str(PROJECT),
            name=RUN_NAME,
            exist_ok=False,
        )
finally:
    restore_flops()
```

## Cell 6 — official resume

Do not parse or reconstruct optimizer, epoch, or scheduler state manually.

```python
from ultralytics import YOLO

last_pt = PROJECT / RUN_NAME / "weights" / "last.pt"
if not last_pt.is_file():
    raise FileNotFoundError(last_pt)
model = YOLO(last_pt)
model.train(resume=True)
```
