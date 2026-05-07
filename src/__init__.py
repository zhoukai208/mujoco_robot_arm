"""Runtime package setup for the robot-arm scripts."""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
