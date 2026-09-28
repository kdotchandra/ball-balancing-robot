"""Tests run without the camera or servos. Run from the project root:

    .venv/bin/python -m pytest test_integrated_system/tests
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # the modules live in test_integrated_system/

DATA = HERE / "data"
# Two real runs copied from logs/ (2026-09-28, tuned profiles): a balance run with the ball pushed away five
# times, and a circle run. Used for replay and regression tests.
BALANCE_RUN = DATA / "run_20260928T140759Z.csv"
CIRCLE_RUN = DATA / "run_20260928T141337Z.csv"
