"""Which way does the ball roll when the plate is tilted? Logs the answer instead of relying on eyes.

Puts the ball under the camera's tracker, tilts the plate with a short, small "doublet" (tilt one
way, then the other, so the ball is left almost at rest) through the same servo mapping the
balance loop uses, and records the ball position throughout. The verdict compares the ball's
motion with the direction the controller assumes:

    +tilt_x  ->  ball moves toward +x      (already confirmed on the real logs)
    +tilt_y  ->  ball moves toward -y      (y is UP in the camera image, so: DOWN on screen)

Without --enable nothing is sent to the servos (dry run: only the ball tracking is checked).
Put the ball near the middle of the plate first, and keep the emergency power cutoff ready.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from ball_tracker import BallTracker

SETTLE_S = 1.2        # plate goes to neutral, ball settles
BASELINE_S = 1.0      # ball tracked with the plate still
HALF_S = 0.6          # duration of each half of the doublet
AFTER_S = 1.2         # plate back at neutral, keep watching
MOVE_MS = 250         # servo transition time for each step (fast, so the pulse is crisp)
MIN_BASELINE_FRAMES = 15
LOG_FIELDS = ["t_s", "phase", "tracked", "x_cm", "y_cm", "tilt_cmd_deg", "servo1", "servo2", "servo3"]


def servo_targets(mapping: np.ndarray, neutral: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                  axis: str, tilt_deg: float) -> np.ndarray:
    """Same conversion as the balance loop and 06_real_tilt_test.py."""
    tilt = np.zeros(2)
    tilt[0 if axis == "x" else 1] = np.deg2rad(tilt_deg)
    q_rad = mapping @ tilt
    proposed = neutral + np.rad2deg(q_rad) * (1000.0 / 240.0)
    return np.clip(np.rint(proposed), lo, hi).astype(int)


def _lagged_double_integral(t: np.ndarray, u: np.ndarray, tau: float, delay: float, dt: float = 0.002) -> np.ndarray:
    """Position response of a unit-gain ball to the command u(t): delay, first-order plate lag, integrate twice."""
    import math
    tg = np.arange(t[0], t[-1] + dt, dt)
    ug = np.interp(tg - delay, t, u, left=0.0)
    uf = np.zeros_like(tg)
    k = 1.0 - math.exp(-dt / tau)
    for i in range(1, len(tg)):
        uf[i] = uf[i - 1] + (ug[i] - uf[i - 1]) * k
    return np.interp(t, tg, np.cumsum(np.cumsum(uf) * dt) * dt)


MODEL_GAIN = 5.886          # KR * g, the ball-on-plate gain the LQR was designed with [m/s^2 per rad]


def analyze(rows: list[dict], axis: str, tilt_deg: float = 2.0) -> dict:
    """Fit the ball's motion with   p = p0 + v0 t + a0 t^2/2 + g * (response to the commanded tilt).

    Unlike comparing peaks, this copes with a ball that is already rolling (v0) and a plate that is
    not perfectly level (a0). g > 0 for x and g < 0 for y is what the controller assumes.
    """
    import itertools
    import math

    expected_sign = +1.0 if axis == "x" else -1.0
    key = "x_cm" if axis == "x" else "y_cm"
    t = np.array([r["t_s"] for r in rows], dtype=float)
    ok = np.array([r["tracked"] for r in rows], dtype=bool)
    pos = np.array([r[key] for r in rows], dtype=float) / 100.0
    u = np.radians(np.array([r["tilt_cmd_deg"] for r in rows], dtype=float))
    base_frames = sum(1 for r in rows if r["phase"] == "baseline" and r["tracked"])
    if base_frames < MIN_BASELINE_FRAMES or not any(r["phase"] == "tilt+" for r in rows):
        return {"verdict": "NO_DATA", "detail": "ball was not tracked well enough before the tilt"}
    after_tilt = [r for r in rows if r["phase"] in ("tilt+", "tilt-", "after") and r["tracked"]]
    if len(after_tilt) < 15:
        return {"verdict": "NO_DATA", "detail": "ball was lost right after the tilt started (it may have rolled off)"}

    t_ok = t[ok]
    best = None
    for tau, delay in itertools.product((0.05, 0.10, 0.20, 0.30), (0.0, 0.05, 0.10, 0.20, 0.30)):
        basis = _lagged_double_integral(t, u, tau, delay)[ok]
        design = np.column_stack([np.ones(len(t_ok)), t_ok - t_ok[0], 0.5 * (t_ok - t_ok[0]) ** 2, basis])
        coef, *_ = np.linalg.lstsq(design, pos[ok], rcond=None)
        resid = pos[ok] - design @ coef
        rms = math.sqrt(float(resid @ resid) / len(resid))
        if best is None or rms < best[0]:
            dof = max(1, len(resid) - 4)
            cov = (float(resid @ resid) / dof) * np.linalg.inv(design.T @ design)
            best = (rms, tau, delay, coef, math.sqrt(float(cov[3, 3])))
    rms, tau, delay, coef, se = best
    g, a0, v0 = float(coef[3]), float(coef[2]), float(coef[1])
    significance = abs(g) / se if se > 0 else float("inf")
    if significance < 4.0 or abs(g) < 0.5:
        verdict = "NO_CLEAR_MOTION"
    elif g * expected_sign > 0:
        verdict = "SIGN_OK"
    else:
        verdict = "SIGN_REVERSED"

    # Model-free cross-check: how far did the ball travel toward / away from the expected side during the
    # first half of the doublet (plus the plate's lag), before the second half can act? Independent of the fit.
    t_plus = next(r["t_s"] for r in rows if r["phase"] == "tilt+")
    base_med = float(np.median([r[key] for r in rows if r["phase"] == "baseline" and r["tracked"]][-10:]))
    win = [(r["t_s"], expected_sign * (r[key] - base_med)) for r in rows
           if r["tracked"] and t_plus + 0.15 <= r["t_s"] <= t_plus + 0.95]
    first_peak_cm = float("nan")
    if len(win) >= 5:
        vals = np.array([v for _t, v in win])
        toward, against = float(vals.max()), float(-vals.min())
        first_peak_cm = toward if toward >= against else -against
        noise_cm = float(np.std([r[key] for r in rows if r["phase"] == "baseline" and r["tracked"]]))
        floor_cm = max(0.15, 4 * noise_cm)
        free = "SIGN_OK" if first_peak_cm > floor_cm else ("SIGN_REVERSED" if first_peak_cm < -floor_cm else "NO_CLEAR_MOTION")
        verdict = free   # the direction comes from the raw motion; the fit is kept only for gain / delay estimates
    return {
        "first_half_peak_cm": first_peak_cm,
        "verdict": verdict, "gain_g_m_s2_per_rad": g, "gain_std_err": se, "significance_g_over_se": significance,
        "expected_sign": expected_sign, "response_vs_healthy": abs(g) / MODEL_GAIN * (1.0 if g * expected_sign > 0 else -1.0),
        "background_accel_m_s2": a0, "initial_velocity_cm_s": v0 * 100.0,
        "plate_lag_tau_s": tau, "delay_s": delay, "fit_rms_mm": rms * 1000.0, "tracked_frames": int(ok.sum()),
    }


def run_sequence(tracker, bus, plan: dict, log_writer=None) -> list[dict]:
    """Drive the phases and record every frame. `bus` may be None (dry run)."""
    rows: list[dict] = []
    neutral = plan["neutral"]
    plus = plan["plus"]
    minus = plan["minus"]
    state = {"servo": neutral.copy(), "tilt": 0.0}

    def send(pos, tilt_deg):
        state["servo"] = pos
        state["tilt"] = tilt_deg
        if bus is not None:
            for servo_id, p in enumerate(pos, start=1):
                bus.move(servo_id, int(p), MOVE_MS)

    def step(phase, duration):
        t_phase = time.perf_counter()
        while time.perf_counter() - t_phase < duration:
            ok, x_m, y_m, _frame, _mask, _dt, _det = tracker.read()
            now = time.perf_counter() - t0
            row = {"t_s": now, "phase": phase, "tracked": bool(ok and x_m is not None),
                   "x_cm": (x_m * 100.0 if ok and x_m is not None else float("nan")),
                   "y_cm": (y_m * 100.0 if ok and y_m is not None else float("nan")),
                   "tilt_cmd_deg": state["tilt"], "servo1": int(state["servo"][0]),
                   "servo2": int(state["servo"][1]), "servo3": int(state["servo"][2])}
            rows.append(row)
            if log_writer is not None:
                log_writer.writerow([f"{row['t_s']:.4f}", row["phase"], int(row["tracked"]),
                                     f"{row['x_cm']:.3f}", f"{row['y_cm']:.3f}", row["tilt_cmd_deg"],
                                     row["servo1"], row["servo2"], row["servo3"]])

    t0 = time.perf_counter()
    send(neutral, 0.0)
    step("settle", SETTLE_S)
    step("baseline", BASELINE_S)
    tracked = sum(1 for r in rows if r["phase"] == "baseline" and r["tracked"])
    if tracked < MIN_BASELINE_FRAMES:
        return rows          # caller reports the problem; nothing was moved yet
    send(plus, plan["tilt_deg"])
    step("tilt+", HALF_S)
    send(minus, -plan["tilt_deg"])
    step("tilt-", HALF_S)
    send(neutral, 0.0)
    step("after", AFTER_S)
    return rows


def run_one(tracker, bus, neutral, mapping, lo, hi, axis: str, tilt_deg: float, log_dir: Path):
    plan = {
        "axis": axis, "tilt_deg": tilt_deg, "neutral": neutral,
        "plus": servo_targets(mapping, neutral, lo, hi, axis, +tilt_deg),
        "minus": servo_targets(mapping, neutral, lo, hi, axis, -tilt_deg),
    }
    log_path = log_dir / f"tilt_test_{axis}_{tilt_deg:g}deg_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.csv"
    print(f"  servo positions: neutral={neutral.tolist()}  +tilt={plan['plus'].tolist()}  -tilt={plan['minus'].tolist()}")
    with open(log_path, "w", newline="", encoding="utf-8", buffering=1) as log_file:
        writer = csv.writer(log_file)
        writer.writerow(LOG_FIELDS)
        rows = run_sequence(tracker, bus, plan, writer)
    baseline_tracked = sum(1 for r in rows if r["phase"] == "baseline" and r["tracked"])
    if baseline_tracked < MIN_BASELINE_FRAMES:
        return {"verdict": "NO_DATA", "detail": f"ball tracked in only {baseline_tracked} baseline frames; plate NOT tilted"}, log_path
    return analyze(rows, axis, tilt_deg), log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Log which way the ball rolls for a small tilt (direction check)")
    parser.add_argument("--port", default="/dev/serial0")
    parser.add_argument("--axis", choices=("x", "y"), default="y")
    parser.add_argument("--tilt-deg", type=float, default=2.0)
    parser.add_argument("--sweep", default="", help="comma list of tilts, e.g. 0.3,0.6,1,2: one doublet each, summary table at the end "
                                                    "(shows whether small tilts still move the ball: a sluggish or dead response at small tilt)")
    parser.add_argument("--tilt-limit-deg", type=float, default=3.0)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--enable", action="store_true", help="really move the servos (otherwise dry run)")
    args = parser.parse_args()
    tilts = [float(v) for v in args.sweep.split(",") if v.strip()] if args.sweep else [args.tilt_deg]
    if any(abs(v) > args.tilt_limit_deg for v in tilts):
        raise SystemExit("Requested tilt exceeds --tilt-limit-deg")

    from servo_calibration.config import load_config, require_values

    limits = load_config("servo_limits.json")
    geometry = load_config("servo_geometry.json")
    mapping_cfg = load_config("servo_mapping.json")
    lo = np.array(require_values(limits, "mechanical_min"), dtype=float)
    hi = np.array(require_values(limits, "mechanical_max"), dtype=float)
    neutral = np.array(require_values(geometry, "neutral_position"), dtype=float).astype(int)
    mapping = np.array([mapping_cfg["axis_mapping"]["tilt_x"], mapping_cfg["axis_mapping"]["tilt_y"]], dtype=float).T
    move = args.enable and args.confirm
    expected = "+x (right on screen)" if args.axis == "x" else "-y (DOWN on screen)"
    print(f"axis={args.axis}  tilts={tilts} deg  (each: +tilt {HALF_S}s, -tilt {HALF_S}s, then neutral)")
    print(f"mapping rows: tilt_x={mapping_cfg['axis_mapping']['tilt_x']}  tilt_y={mapping_cfg['axis_mapping']['tilt_y']}")
    print(f"limits: min={lo.astype(int).tolist()} max={hi.astype(int).tolist()}")
    print(f"expected if the mapping is right: ball moves toward {expected} during the +tilt half")
    print("DRY RUN: no servo command will be sent." if not move else "REAL RUN: servos WILL move.")

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tracker = BallTracker(camera_index=args.camera, mirror=True, detector_mode="old")
    bus = None
    results: list[tuple[float, dict, Path]] = []
    try:
        if move:
            from servo_calibration.hx35h import HX35HBus
            bus = HX35HBus(args.port)
        for k, tilt in enumerate(tilts):
            if move:
                input(f"\n[{k + 1}/{len(tilts)}] tilt {tilt:g} deg: ball near the plate centre? cutoff ready? Enter to start, Ctrl+C to abort: ")
            else:
                print(f"\n[{k + 1}/{len(tilts)}] tilt {tilt:g} deg")
            result, path = run_one(tracker, bus, neutral, mapping, lo, hi, args.axis, tilt, log_dir)
            results.append((tilt, result, path))
            print(f"  -> {result['verdict']}   log: {path.name}")
            if result["verdict"] == "NO_DATA":
                print(f"     {result.get('detail', '')}")
    except KeyboardInterrupt:
        print("\ninterrupted -- returning the plate to neutral")
    finally:
        try:
            if bus is not None:
                for servo_id, p in enumerate(neutral, start=1):
                    bus.move(servo_id, int(p), 500)
                time.sleep(0.6)
                bus.close()
        finally:
            tracker.release()

    print("\n================ RESULT ================")
    print(f"{'tilt deg':>9}{'verdict':>17}{'1st-half cm':>12}{'g m/s2/rad':>12}{'+/-':>7}{'% of model':>12}{'lag tau s':>10}{'delay s':>9}{'fit rms mm':>11}")
    for tilt, r, _path in results:
        if "gain_g_m_s2_per_rad" not in r:
            print(f"{tilt:>9g}{r['verdict']:>17}")
            continue
        print(f"{tilt:>9g}{r['verdict']:>17}{r['first_half_peak_cm']:>+12.2f}{r['gain_g_m_s2_per_rad']:>+12.2f}{r['gain_std_err']:>7.2f}"
              f"{100 * r['response_vs_healthy']:>+11.0f}%{r['plate_lag_tau_s']:>10.2f}{r['delay_s']:>9.2f}{r['fit_rms_mm']:>11.1f}")
    pk = [(t_, r["first_half_peak_cm"]) for t_, r, _p in results if "first_half_peak_cm" in r and r["first_half_peak_cm"] == r["first_half_peak_cm"]]
    big = [(t_, v) for t_, v in pk if t_ >= 0.5]
    if len(big) >= 3:
        slope, icpt = np.polyfit([t_ for t_, _v in big], [v for _t2, v in big], 1)
        if slope > 0:
            print(f"\nFirst-half travel vs tilt (tilts >= 0.5 deg): {slope:.2f} cm per degree, reaching zero at about "
                  f"{-icpt / slope:.2f} deg -> that is the dead zone (backlash / sticking) of this axis.")
    ok = [r for _t, r, _p in results if "gain_g_m_s2_per_rad" in r]
    if len(ok) >= 2:
        big = max(ok, key=lambda r: abs(r["gain_g_m_s2_per_rad"]))
        small = ok[0]
        ratio = abs(small["gain_g_m_s2_per_rad"]) / max(abs(big["gain_g_m_s2_per_rad"]), 1e-9)
        if ratio < 0.5:
            print(f"\nSmall tilts move the ball far less than large ones (gain at the smallest tilt is {100 * ratio:.0f}% of the best): "
                  "the plate/servo/ball has a dead zone or sticks at small commands.")
        else:
            print("\nThe gain is about the same at small and large tilts: no dead zone showing up in this range.")
    print("(dry run: the plate was not moved, so results here only test the tracking)" if not move else "")


if __name__ == "__main__":
    main()
