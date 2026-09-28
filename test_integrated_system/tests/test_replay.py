"""Regression tests on real run logs: the controller maths and the scoring must reproduce what was logged."""

import csv
import json
import math

import numpy as np
import pytest

import run_metrics as RM
from conftest import BALANCE_RUN, CIRCLE_RUN, DATA
from lqr_controller import LQRController
from path_tuning import select_gain


def _rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("run", [BALANCE_RUN, CIRCLE_RUN], ids=["balance", "circle"])
def test_lqr_command_replays_the_log(run):
    """Recompute theta_x/y_cmd from the logged state with the logged gains; it must match the log.

    The loop computes the command from theta_actual as it was BEFORE this row's update, i.e. the value logged
    in the previous row. Rows at the tilt limit (clipped twice in main.py) and the first valid row are skipped.
    """
    meta = json.loads(run.with_suffix(".json").read_text(encoding="utf-8"))
    k_main = np.array(meta["k_effective"], float)
    k_path = np.array(meta["path_k_full"], float) if meta.get("path_k_full") else None
    ctrl = LQRController(k_axis=k_main, tilt_limit_rad=math.radians(8.0))
    rows = _rows(run)
    f = lambda r, k: float(r[k])
    checked, worst = 0, 0.0
    for prev, row in zip(rows, rows[1:]):
        if row["control_valid"] != "1" or prev["control_valid"] != "1":
            continue
        want_x, want_y = f(row, "theta_x_cmd_deg"), f(row, "theta_y_cmd_deg")
        if max(abs(want_x), abs(want_y)) > 7.9:
            continue
        ctrl.k_axis = select_gain(k_main, k_path, int(f(row, "path_phase")))
        tx, ty = ctrl.compute(
            x_m=f(row, "x_m"), y_m=f(row, "y_m"), x_dot=f(row, "x_dot_mps"), y_dot=f(row, "y_dot_mps"),
            theta_x_actual=math.radians(f(prev, "theta_x_actual_deg")),
            theta_y_actual=math.radians(f(prev, "theta_y_actual_deg")),
            x_ref=f(row, "x_ref_m"), y_ref=f(row, "y_ref_m"),
            x_ref_dot=f(row, "x_ref_dot_mps"), y_ref_dot=f(row, "y_ref_dot_mps"),
        )
        worst = max(worst, abs(math.degrees(tx) - want_x), abs(math.degrees(ty) - want_y))
        checked += 1
    assert checked > 300
    assert worst < 0.01, f"controller output differs from the log by up to {worst:.4f} deg"


# Values printed by run_metrics.score_run when these logs were copied in (2026-09-28).
EXPECTED = {
    BALANCE_RUN: {"lost": False, "settle_s": 0.762, "steady_mm": 21.764, "rms_mm": 22.5568, "overshoot_pct": 68.9747},
    CIRCLE_RUN: {"lost": False, "settle_s": 0.6596, "steady_mm": 14.6405, "rms_mm": 16.6327,
                 "overshoot_pct": 27.3644, "track_mm": 13.0757},
}


@pytest.mark.parametrize("run", [BALANCE_RUN, CIRCLE_RUN], ids=["balance", "circle"])
def test_run_metrics_scores_are_stable(run):
    got = RM.score_run(run, RM.default_zone_cm())
    for key, value in EXPECTED[run].items():
        if isinstance(value, bool):
            assert got[key] == value
        else:
            assert got[key] == pytest.approx(value, abs=1e-3), key


def test_analyze_runs_on_test_data():
    import analyze_runs as AR
    results, skipped = AR.collect(["demo_balance_20260928_210754", "demo_circle_20260928_211330"], DATA)
    assert not skipped and len(results) == 2
    by_mode = {r["mode"]: r for r in results}
    assert by_mode["path"]["error_cm"] == pytest.approx(1.31, abs=0.01)
    assert not by_mode["path"]["limit_cycle"]
    assert by_mode["balance"]["end"] == "completed"
    assert "tag" in AR.report(results, ["demo_balance_20260928_210754", "demo_circle_20260928_211330"])
