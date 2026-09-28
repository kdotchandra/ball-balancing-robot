"""Score run logs against the project's evaluation table.

    python run_metrics.py logs/run_A.csv logs/run_B.csv ...

Every target comes from `theory_limits.py`, which derives it either from the linear theory the LQR
was designed on or from a measured hardware limit -- see that file for the derivations. This module
only measures the logs and compares. For the Thai write-up (comparison, analysis, discussion) use
`metrics_report.py`, which builds on this one.

Only rows with a valid ball detection are used, and the distance is always to the reference at that
instant, so the same code scores a balance run (reference fixed at the centre) and a path run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import theory_limits as TH

HOLD_INSIDE_S = 2.0        # the ball must stay in the centre zone this long to count as settled
STEADY_WINDOW_S = 5.0      # steady-state error is averaged over the last part of the run
RMS_WINDOW_S = 15.0        # RMS is taken over the last stretch, after the transient
MIN_RUN_S = 8.0
MIN_OVERSHOOT_START_MM = 20.0   # below half a ball diameter the overshoot ratio is meaningless
PHASE_HOLD, PHASE_PATH = 1, 3
REFERENCE_PATH_AMP_MM = 30.0    # the 3 cm path the tracking target is stated for; larger paths are scaled to it

CONFIG_PATH = Path(__file__).resolve().parent / "servo_calibration" / "config" / "control_config.json"


def default_zone_cm() -> float:
    """Centre zone used for settling: `reference_radius_cm`, which is one ball diameter (4 cm)."""
    try:
        return float(json.loads(CONFIG_PATH.read_text()).get("reference_radius_cm", 4.0))
    except Exception:
        return 4.0


def load_run(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("empty log")

    def col(name: str, default: float = float("nan")) -> np.ndarray:
        return np.array([float(r[name]) if r.get(name, "") not in ("", None) else default for r in rows])

    data = {name: col(name) for name in ("t_s", "control_valid", "x_m", "y_m", "x_ref_m", "y_ref_m",
                                         "lost_neutralized")}
    data["path_phase"] = col("path_phase", 0.0) if "path_phase" in rows[0] else np.zeros(len(rows))
    return data


def _settle_time(t: np.ndarray, err_mm: np.ndarray, zone_mm: float, t0: float) -> float:
    """Earliest time (measured from t0) after which the error stays inside the zone for 2 s."""
    inside = err_mm <= zone_mm
    for i in range(len(t)):
        if not inside[i]:
            continue
        if t[-1] < t[i] + HOLD_INSIDE_S:
            break
        if inside[(t >= t[i]) & (t <= t[i] + HOLD_INSIDE_S)].all():
            return float(t[i] - t0)
    return float("nan")


def _overshoot_pct(ex: np.ndarray, ey: np.ndarray) -> float:
    """How far the ball went past the reference, as a percentage of where it started.

    The initial error vector defines the approach direction; the ball's position is projected onto
    it, so a negative projection means it crossed the reference. This is the step-response overshoot
    of that one axis of motion.
    """
    start = np.array([ex[:3].mean(), ey[:3].mean()])
    d0 = float(np.linalg.norm(start))
    if d0 < MIN_OVERSHOOT_START_MM:   # started too close to the reference for the ratio to mean anything
        return float("nan")
    unit = start / d0
    projection = ex * unit[0] + ey * unit[1]
    return float(max(0.0, -projection.min()) / d0 * 100.0)


def _amplitude_ratio(ball_mm: np.ndarray, ref_mm: np.ndarray) -> float:
    """How much of the reference's swing the ball actually performed, on one axis."""
    span = float(np.ptp(ref_mm))
    return float(np.ptp(ball_mm) / span) if span > 1.0 else float("nan")


def _band_rms(t: np.ndarray, v_mm: np.ndarray, lo_hz: float = 0.2, hi_hz: float = 2.0) -> float:
    """RMS of the 0.2-2 Hz motion of one axis after removing the slow trend (a cubic fit).

    The lap frequency is 0.05 Hz, so this band is where hunting and oscillation show up; it is the
    quantity the integral-gain ladder must not let grow.
    """
    if len(t) < 60:
        return float("nan")
    dt = float(np.median(np.diff(t)))
    grid = np.arange(t[0], t[-1], dt)
    v = np.interp(grid, t, v_mm)
    v = v - np.polyval(np.polyfit(grid, v, 3), grid)
    spec = np.abs(np.fft.rfft(v))
    freq = np.fft.rfftfreq(len(v), dt)
    band = (freq >= lo_hz) & (freq <= hi_hz)
    return float(np.sqrt(2.0 * np.sum(spec[band] ** 2)) / len(v))


def _fundamental_gain(t: np.ndarray, ball_mm: np.ndarray, ref_mm: np.ndarray) -> tuple[float, float]:
    """(gain, phase in degrees) of the ball relative to the reference at the lap frequency.

    Peak-to-peak tells how far the ball swung; this tells how much of that swing was actually in
    step with the reference. Both the ball and the reference are projected onto cos/sin at the
    reference's own dominant frequency, so it works for the hexagon (fundamental) as well.
    """
    if len(t) < 60:
        return float("nan"), float("nan")
    dt = float(np.median(np.diff(t)))
    grid = np.arange(t[0], t[-1], dt)
    ref_u = np.interp(grid, t, ref_mm)
    spec = np.abs(np.fft.rfft((ref_u - ref_u.mean()) * np.hanning(len(grid))))
    freqs = np.fft.rfftfreq(len(grid), dt)
    band = freqs > 0.5 / (t[-1] - t[0])
    if not band.any() or spec[band].max() <= 0:
        return float("nan"), float("nan")
    w = 2.0 * np.pi * float(freqs[band][np.argmax(spec[band])])
    basis = np.stack([np.cos(w * t), np.sin(w * t), np.ones_like(t)], axis=1)
    cb = np.linalg.lstsq(basis, ball_mm, rcond=None)[0]
    cr = np.linalg.lstsq(basis, ref_mm, rcond=None)[0]
    zb, zr = complex(cb[0], -cb[1]), complex(cr[0], -cr[1])
    if abs(zr) < 1e-9:
        return float("nan"), float("nan")
    h = zb / zr
    return float(abs(h)), float(np.degrees(np.angle(h)))


def score_run(path: Path, zone_cm: float) -> dict:
    d = load_run(Path(path))
    valid = d["control_valid"] == 1
    out: dict = {"file": Path(path).name, "n_valid": int(valid.sum())}
    if valid.sum() < 10 or d["t_s"][valid][-1] - d["t_s"][valid][0] < MIN_RUN_S:
        out.update(kind="short", success=False, note="too short / ball barely tracked")
        return out

    t = d["t_s"][valid]
    ex = (d["x_m"][valid] - d["x_ref_m"][valid]) * 1000.0
    ey = (d["y_m"][valid] - d["y_ref_m"][valid]) * 1000.0
    err = np.hypot(ex, ey)
    phase = d["path_phase"][valid]
    t0 = float(t[0])
    is_path = bool(np.any(phase == PHASE_PATH))
    meta = read_meta(Path(path)) or {}
    out.update(kind="path" if is_path else "balance",
               duration_s=float(t[-1] - t0),
               lost=bool(np.any(d["lost_neutralized"] == 1)) or meta.get("ended_reason") == "ball_lost",
               ended_reason=meta.get("ended_reason", ""),
               start_cm=float(err[:5].mean() / 10.0))

    # Balance metrics: the whole run for a balance run, the hold phase for a path run.
    sel = (phase == PHASE_HOLD) if is_path else np.ones(len(t), dtype=bool)
    ts, es = t[sel], err[sel]
    if len(ts) > 3:
        st = _settle_time(ts, es, zone_cm * 10.0, t0)
        out["settle_s"] = st
        out["settle_2cm_s"] = _settle_time(ts, es, 20.0, t0)
        out["steady_mm"] = float(es[ts >= ts[-1] - STEADY_WINDOW_S].mean())
        after = ts >= max(ts[-1] - RMS_WINDOW_S, t0 + st) if not math.isnan(st) else np.zeros(len(ts), bool)
        out["rms_mm"] = float(math.sqrt(np.mean(es[after] ** 2))) if after.any() else float("nan")
        out["overshoot_pct"] = _overshoot_pct(ex[sel], ey[sel])
    else:
        out.update(settle_s=float("nan"), settle_2cm_s=float("nan"), steady_mm=float("nan"),
                   rms_mm=float("nan"), overshoot_pct=float("nan"))

    if is_path:
        p = phase == PHASE_PATH
        out["track_mm"] = float(err[p].mean())
        out["track_rms_mm"] = float(math.sqrt(np.mean(err[p] ** 2)))
        out["track_max_mm"] = float(err[p].max())
        out["track_x_mm"] = float(np.abs(ex[p]).mean())
        out["track_y_mm"] = float(np.abs(ey[p]).mean())
        bx, by = d["x_m"][valid][p] * 1000.0, d["y_m"][valid][p] * 1000.0
        rfx, rfy = d["x_ref_m"][valid][p] * 1000.0, d["y_ref_m"][valid][p] * 1000.0
        ratios = [r for r in (_amplitude_ratio(bx, rfx), _amplitude_ratio(by, rfy)) if r == r]
        out["amp_ratio"] = float(np.mean(ratios)) if ratios else float("nan")
        out["ball_amp_mm"] = float(np.mean([np.ptp(bx), np.ptp(by)]) / 2.0)
        out["ref_amp_mm"] = float(np.mean([np.ptp(rfx), np.ptp(rfy)]) / 2.0)
        gains = [_fundamental_gain(t[p], b, r) for b, r in ((bx, rfx), (by, rfy))]
        in_step = [g for g, _ph in gains if g == g]
        lags = [ph for g, ph in gains if ph == ph]
        out["gain_in_step"] = float(np.mean(in_step)) if in_step else float("nan")
        out["phase_deg"] = float(np.mean(lags)) if lags else float("nan")
        # Tracking is judged relative to the path size: a bigger circle is not held to the same millimetres.
        scale = REFERENCE_PATH_AMP_MM / out["ref_amp_mm"] if out["ref_amp_mm"] > 1.0 else 1.0
        out["track_mm_scaled"] = out["track_mm"] * scale
        rb, rr = np.hypot(bx, by), np.hypot(rfx, rfy)
        out["radius_ratio"] = float(rb.mean() / rr.mean()) if rr.mean() > 1.0 else float("nan")
        ball_ang = np.degrees(np.unwrap(np.arctan2(by, bx)))
        ref_ang = np.degrees(np.unwrap(np.arctan2(rfy, rfx)))
        diff = (ball_ang - ref_ang + 180.0) % 360.0 - 180.0
        out["angle_ok_pct"] = float(100.0 * np.mean(np.abs(diff) < 45.0))
        out["net_angle_deg"], out["ref_net_angle_deg"] = float(ball_ang[-1] - ball_ang[0]), float(ref_ang[-1] - ref_ang[0])
        out["path_seconds"] = float(t[p][-1] - t[p][0])
        out["osc_rms_mm"] = float(np.mean([_band_rms(t[p], bx), _band_rms(t[p], by)]))
        out["track_ok"] = out["track_mm_scaled"] <= _limit("track_mm")
        # Path trials: "success" is reliability -- the run went the whole way without losing the ball.
        # How well it followed the path is reported separately (track_mm, radius_ratio, track_ok).
        out["success"] = not out["lost"]
    else:
        st = out["settle_s"]
        out["success"] = (not out["lost"]) and (not math.isnan(st)) and st <= _limit("settle_s")
    return out


# ---------------------------------------------------------------------------------- run sidecars
LOG_DIR = Path(__file__).resolve().parent / "logs"
COUNTED_END_REASONS = ("completed", "ball_lost")   # aborted (Ctrl+C) and never-tracked trials are not counted


def read_meta(csv_path: Path) -> dict | None:
    side = Path(csv_path).with_suffix(".json")
    if not side.exists():
        return None
    try:
        return json.loads(side.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def collect_runs(tag: str, log_dir: Path | None = None) -> list[Path]:
    """CSV paths of the counted trials with this tag, oldest first.

    Counted = the run ended by itself (`completed`) or the ball was lost (`ball_lost`, a genuine failure).
    Not counted: `never_tracked` (the ball was never seen: a set-up problem) and `interrupted` (the user
    stopped it). Those logs are kept, they are just not results.
    """
    directory = log_dir or LOG_DIR
    found = []
    for side in sorted(directory.glob("run_*.json")):
        meta = read_meta(side.with_suffix(".csv"))
        if meta and meta.get("tag") == tag and meta.get("ended_reason") in COUNTED_END_REASONS:
            found.append(side.with_suffix(".csv"))
    return found


LADDER_BASELINE_TAG = "circle"      # kI = 0 reference for the ladder (the standard circle trials)
# before any tagged circle trials exist, the two earlier balance-gain circle runs are the kI = 0 baseline
LADDER_LEGACY_BASELINE = ("run_20260924T082420Z.csv", "run_20260924T082540Z.csv")


def ladder_summary(tags: list[str], baseline: list[Path] | None = None) -> list[dict]:
    """Apply the pre-registered rule for choosing the integral gain kI from the ladder trials.

    Rule (fixed before the trials): pick the largest kI that (a) loses the ball in no run, (b) keeps the
    0.2-2 Hz ball motion within 1.5x of the baseline, and (c) has a mean orbit-radius ratio AND a mean
    tracking error better than the baseline; if none passes, kI = 0. The baseline is the kI = 0 circle
    trials. Two runs per value only detect gross failure, they do not rank small differences.
    """
    zone = default_zone_cm()
    base_runs = [score_run(p, zone) for p in (baseline or [])]
    base_runs = [r for r in base_runs if r.get("kind") == "path"]

    def mean(rs, key):
        vals = [r[key] for r in rs if key in r and r[key] == r[key]]
        return float(np.mean(vals)) if vals else float("nan")

    base = {"n": len(base_runs), "ki": 0.0, "radius": mean(base_runs, "radius_ratio"),
            "track": mean(base_runs, "track_mm"), "osc": mean(base_runs, "osc_rms_mm")}
    rows = []
    for tag in tags:
        paths = collect_runs(tag)
        if not paths:
            continue
        runs = [score_run(p, zone) for p in paths]
        meta = read_meta(paths[0]) or {}
        rs = [r for r in runs if r.get("kind") == "path"]
        row = {"tag": tag, "ki": float(meta.get("path_ki", float("nan"))), "n": len(runs),
               "lost": sum(1 for r in runs if r.get("lost")), "radius": mean(rs, "radius_ratio"),
               "track": mean(rs, "track_mm"), "osc": mean(rs, "osc_rms_mm")}
        row["pass_lost"] = row["lost"] == 0
        row["pass_osc"] = row["osc"] <= 1.5 * base["osc"] if base["osc"] == base["osc"] else False
        row["pass_better"] = row["radius"] > base["radius"] and row["track"] < base["track"]
        row["passes"] = row["pass_lost"] and row["pass_osc"] and row["pass_better"]
        rows.append(row)
    rows.sort(key=lambda r: r["ki"])
    chosen = max((r["ki"] for r in rows if r["passes"]), default=0.0)
    return [base] + rows + [{"chosen_ki": chosen}]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for a success rate. With n = 10 it is wide, and it should be shown."""
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


_TARGETS = {t.key: t for t in TH.targets()}


def _limit(key: str) -> float:
    return _TARGETS[key].limit


def worst(results: list[dict], key: str) -> float:
    vals = [r[key] for r in results if key in r and isinstance(r[key], float) and not math.isnan(r[key])]
    return max(vals) if vals else float("nan")


def aggregate(results: list[dict]) -> dict:
    """Worst case of each metric plus the success rate, ready to compare with the targets."""
    scored = [r for r in results if r.get("kind") != "short"]
    agg = {key: worst(scored, key) for key in ("rms_mm", "settle_s", "steady_mm", "overshoot_pct")}
    # tracking is compared on the 3 cm scale (see REFERENCE_PATH_AMP_MM) so paths of different size mix fairly
    agg["track_mm"] = worst(scored, "track_mm_scaled")
    agg["n_runs"] = len(scored)
    agg["n_success"] = sum(1 for r in scored if r["success"])
    agg["success"] = 100.0 * agg["n_success"] / len(scored) if scored else float("nan")
    return agg


def _one_line(r: dict) -> str:
    """One readable line for a finished trial (used by run_experiment.sh after each trial)."""
    if r.get("kind") == "short":
        return f"{r['file']}: too short to score ({r.get('note', '')})"
    head = f"{r['file']}: start {r['start_cm']:.1f} cm, "
    if r["kind"] == "balance":
        body = (f"entered the 4 cm zone after {fmt_num(r['settle_s'])} s, steady-state {fmt_num(r['steady_mm'])} mm, "
                f"RMS {fmt_num(r['rms_mm'])} mm")
    else:
        body = (f"tracking error {r['track_mm']:.1f} mm, orbit radius {r['radius_ratio']:.2f} x commanded, "
                f"direction match {r['angle_ok_pct']:.0f}%")
    return head + body + (" -- BALL LOST" if r["lost"] else " -- ok")


def fmt_num(value: float) -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.1f}"


def _flag(value: float, limit: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return " n/a"
    return "  ok" if value <= limit else "FAIL"


def print_summary(paths: list[Path], reference_radius_cm: float | None = None) -> list[dict]:
    zone = reference_radius_cm if reference_radius_cm is not None else default_zone_cm()
    print("\n================ RUN METRICS ================")
    print(f"centre zone for settling: radius {zone:g} cm (control_config.json = one ball diameter), "
          f"must stay inside {HOLD_INSIDE_S:g} s")
    results = []
    for path in paths:
        try:
            results.append(score_run(Path(path), zone))
        except Exception as exc:
            print(f"{Path(path).name}: cannot read ({exc})")
    if not results:
        return results
    print(f"{'run':<26}{'type':>8}{'start':>7}{'settle s':>10}{'(2cm)':>8}{'RMS mm':>8}"
          f"{'steady':>8}{'over %':>8}{'track mm':>10}{'amp':>6}  result")
    nan = float("nan")
    for r in results:
        if r["kind"] == "short":
            print(f"{r['file']:<26}{'short':>8}   {r['note']}")
            continue
        amp = r.get("amp_ratio", nan)
        print(f"{r['file']:<26}{r['kind']:>8}{r['start_cm']:>6.1f}c"
              f"{r['settle_s']:>10.1f}{r['settle_2cm_s']:>8.1f}{r['rms_mm']:>8.1f}{r['steady_mm']:>8.1f}"
              f"{r['overshoot_pct']:>8.0f}{r.get('track_mm', nan):>10.1f}"
              f"{(amp * 100 if amp == amp else nan):>5.0f}%  {'PASS' if r['success'] else 'FAIL'}"
              f"{' (ball lost)' if r['lost'] else ''}")

    agg = aggregate(results)
    if not agg["n_runs"]:
        return results
    print("\ntargets (source in theory_limits.py), worst run of each metric:")
    print(f"{'metric':<24}{'worst':>9}{'target':>10}{'theory':>10}  verdict")
    for key in ("rms_mm", "settle_s", "steady_mm", "overshoot_pct", "track_mm"):
        tgt = _TARGETS[key]
        th = tgt.theory_value
        print(f"{tgt.metric:<24}{agg[key]:>9.1f}{tgt.limit:>10.1f}"
              f"{(th if th is not None else nan):>10.2f}  {_flag(agg[key], tgt.limit)}")
    lo, hi = wilson_interval(agg["n_success"], agg["n_runs"])
    print(f"{'success rate':<24}{agg['success']:>8.0f}%{_limit('success'):>9.0f}%{'':>10}  "
          f"{'  ok' if agg['success'] >= _limit('success') else 'FAIL'}"
          f"   ({agg['n_success']}/{agg['n_runs']} runs, 95% interval {100 * lo:.0f}-{100 * hi:.0f}%)")
    print("  balance run passes: the ball was never lost and it entered the centre zone within the target time."
          "\n  path run passes: it went the whole way without losing the ball (tracking accuracy is judged by its own row).")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Score run logs against the evaluation table")
    parser.add_argument("logs", nargs="*", type=Path)
    parser.add_argument("--zone-cm", type=float, default=None,
                        help="centre-zone radius for settling (default: from control_config.json)")
    parser.add_argument("--theory", action="store_true", help="also print the derivations from theory_limits.py")
    parser.add_argument("--tag", default=None, help="score every counted run whose sidecar carries this tag")
    parser.add_argument("--count-tag", default=None, help="print how many counted runs carry this tag, and exit")
    parser.add_argument("--last-of-tag", default=None, help="one-line result of the newest counted run of this tag")
    parser.add_argument("--ladder", nargs="+", default=None, metavar="TAG",
                        help="apply the integral-gain selection rule to these ladder tags (baseline = the circle tag)")
    args = parser.parse_args()
    if args.count_tag is not None:
        print(len(collect_runs(args.count_tag)))
        return
    if args.last_of_tag is not None:
        runs = collect_runs(args.last_of_tag)
        if runs:
            r = score_run(runs[-1], args.zone_cm if args.zone_cm is not None else default_zone_cm())
            print(_one_line(r))
        return
    if args.ladder:
        baseline = collect_runs(LADDER_BASELINE_TAG) or [LOG_DIR / n for n in LADDER_LEGACY_BASELINE]
        rows = ladder_summary(args.ladder, baseline=baseline)
        base, chosen = rows[0], rows[-1]["chosen_ki"]
        print(f"baseline kI=0 ({base['n']} runs of tag '{LADDER_BASELINE_TAG}'): radius ratio {base['radius']:.2f}, "
              f"tracking {base['track']:.1f} mm, 0.2-2 Hz motion {base['osc']:.2f} mm")
        print(f"{'tag':<18}{'kI':>6}{'runs':>5}{'lost':>5}{'radius':>8}{'track mm':>9}{'osc mm':>8}   lost? osc<=1.5x? better?  -> pass")
        for r in rows[1:-1]:
            print(f"{r['tag']:<18}{r['ki']:>6.2f}{r['n']:>5}{r['lost']:>5}{r['radius']:>8.2f}{r['track']:>9.1f}{r['osc']:>8.2f}"
                  f"   {'ok ' if r['pass_lost'] else 'NO '}  {'ok ' if r['pass_osc'] else 'NO '}         {'ok ' if r['pass_better'] else 'NO '}  -> {'PASS' if r['passes'] else 'no'}")
        print(f"\nchosen kI by the pre-registered rule: {chosen:g}")
        return
    if args.theory:
        for line in TH.summary_lines():
            print(line)
    logs = collect_runs(args.tag) if args.tag else args.logs
    if not logs:
        parser.error("give log files, or --tag NAME with at least one counted run")
    print_summary(logs, reference_radius_cm=args.zone_cm)


if __name__ == "__main__":
    main()
