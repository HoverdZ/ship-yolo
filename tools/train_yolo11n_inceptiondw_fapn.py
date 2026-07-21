"""Colab-ready formal training entrypoint for YOLO11n-InceptionDW-FaPN."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fapn_training import run_training


if __name__ == "__main__":
    run_training("inceptiondw")
