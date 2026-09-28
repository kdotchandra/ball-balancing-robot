"""Compare groups of runs by tag -- the analysis used while tuning the demos (2026-09-27/28).

run_metrics.py scores runs against the report's targets; this tool answers the tuning questions:
is one setting better than another run for run, did anything fall into a limit cycle, how much do the
servos move, and where was the ball lost. Every run with one of the given tags is included (interrupted
and lost runs too, flagged), unreadable logs are skipped with a note.

    python analyze_runs.py TAG [TAG ...] [--after STAMP] [--before STAMP] [--log-dir DIR]

The first tag is the reference: each other tag gets a permutation test against it (mean tracking error
for path runs, mean distance from the centre over the last 10 s for balance runs). With interleaved runs
(A, B, A, B, ...) that comparison is fair; otherwise the plate's condition drifts over time.

Per run:
  path runs     tracking error (mean, p95) in the path phase, ball/reference radius, limit cycle
  balance runs  start position, time until the ball stays within 2 cm for 1 s, last-10 s distance,
                farthest excursion after first reaching 1 cm, PASS (no loss, 2 cm within 5 s, last-10 s < 1 cm)
  both          servo jitter (servo 1 command, high-frequency part), % time the ball is still (< 0.3 cm/s),
                detection drop-outs (and how many in the dark left region of the developer's room, x < -4.5 cm),
                and for a lost ball where and how fast it went
A limit cycle is flagged when the tracking error's spectrum peaks above 0.4 Hz with >= 60 % of its power
within +-0.05 Hz of the peak -- the signature of the 0.6-0.8 Hz oscillations seen with too much filter lag
or dead-zone lead.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import run_metrics as RM

FPS = 30.0
DARK_LEFT_X_CM = -4.5
LIMIT_CYCLE_MIN_HZ = 0.4
LIMIT_CYCLE_MIN_SHARE = 0.6


def _col(rows: list[dict], name: str) -> np.ndarray:
    return np.array([float(r[name]) if r.get(name, "") not in ("", None) else np.nan for r in rows])


def _highpass(v: np.ndarray, n: int = 9) -> np.ndarray:
    return v - np.convolve(v, np.ones(n) / n, "same")


def _speed_cms(t: np.ndarray, x_cm: np.ndarray, y_cm: np.ndarray) -> np.ndarray:
    k = np.ones(5) / 5
    return np.hypot(np.gradient(np.convolve(x_cm, k, "same"), t), np.gradient(np.convolve(y_cm, k, "same"), t))


def limit_cycle(t: np.ndarray, err_cm: np.ndarray) -> tuple[float, float, bool]:
    """(peak frequency Hz, share of power within +-0.05 Hz of it, flagged) of an error signal."""
    if len(t) < 150:
        return math.nan, math.nan, False
    tu = np.arange(t[0], t[-1], 1 / FPS)
    e = np.interp(tu, t, err_cm)
    e = e - e.mean()
    power = np.abs(np.fft.rfft(e * np.hanning(len(e)))) ** 2
    freq = np.fft.rfftfreq(len(e), 1 / FPS)
    band = freq > 0.08
    peak = float(freq[band][np.argmax(power[band])])
    share = float(power[band][(freq[band] > peak - 0.05) & (freq[band] < peak + 0.05)].sum() / power[band].sum())
    return peak, share, peak > LIMIT_CYCLE_MIN_HZ and share >= LIMIT_CYCLE_MIN_SHARE


def analyze_run(csv_path: Path) -> dict | None:
    """Metrics of one run, or None if its log cannot be read."""
    meta = RM.read_meta(csv_path)
    if meta is None or b"\x00" in Path(csv_path).read_bytes():
        return None
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 30:
        return None
    t = _col(rows, "t_s")
    det = _col(rows, "detected") > 0.5
    phase = _col(rows, "path_phase")
    x, y = _col(rows, "x_m") * 100, _col(rows, "y_m") * 100
    xr, yr = _col(rows, "x_ref_m") * 100, _col(rows, "y_ref_m") * 100
    speed = _speed_cms(t, x, y)
    out = {"stamp": Path(csv_path).stem[4:], "tag": meta.get("tag", ""), "mode": meta.get("mode", ""),
           "end": meta.get("ended_reason", ""), "detected_pct": 100 * det.mean()}
    active = det & ((phase >= 2) if out["mode"] == "path" else np.ones_like(det))
    out["servo_jitter"] = float(np.std(_highpass(_col(rows, "servo1"))[active])) if active.sum() > 30 else math.nan
    out["still_pct"] = float(100 * np.mean(speed[active] < 0.3)) if active.any() else math.nan

    drops = np.flatnonzero((np.diff(det.astype(int)) == -1) & (phase[1:] >= (1 if out["mode"] == "path" else 0)))
    out["drops"] = int(len(drops))
    out["drops_dark"] = int(np.sum(x[drops] < DARK_LEFT_X_CM))
    out["loss"] = ""
    if out["end"] == "ball_lost" and det.any():
        i = int(np.flatnonzero(det)[-1])
        j = max(0, i - 6)
        v = math.hypot(x[i] - x[j], y[i] - y[j]) / max(1e-3, t[i] - t[j])
        where = "dark left" if x[i] < DARK_LEFT_X_CM and v < 15 else ("fast exit" if v > 15 else "other")
        out["loss"] = f"{where} ({x[i]:+.1f},{y[i]:+.1f}) cm at {v:.0f} cm/s, t={t[i]:.1f} s"

    if out["mode"] == "path":
        p = det & (phase == 3)
        err = np.hypot(x - xr, y - yr)
        out["error_cm"] = float(err[p].mean()) if p.sum() >= 60 else math.nan
        out["p95_cm"] = float(np.percentile(err[p], 95)) if p.sum() >= 60 else math.nan
        out["ball_r_cm"] = float(np.hypot(x[p], y[p]).mean()) if p.any() else math.nan
        out["ref_r_cm"] = float(np.hypot(xr[p], yr[p]).mean()) if p.any() else math.nan
        out["peak_hz"], out["peak_share"], out["limit_cycle"] = limit_cycle(t[p], (x - xr)[p])
        out["key"] = out["error_cm"]
    else:
        dist = np.hypot(x, y)
        i0 = int(np.flatnonzero(det)[0]) if det.any() else 0
        out["start_cm"] = (float(x[i0]), float(y[i0]))
        out["t_to_2cm_s"] = math.nan
        for i in np.flatnonzero(det & (dist < 2.0)):
            w = (t >= t[i]) & (t < t[i] + 1.0)
            if det[w].all() and (dist[w] < 2.0).all():
                out["t_to_2cm_s"] = float(t[i] - t[i0])
                break
        last = det & (t > t[-1] - 10)
        out["last10_cm"] = float(dist[last].mean()) if last.any() else math.nan
        first_in = np.flatnonzero(det & (dist < 1.0))
        out["overshoot_cm"] = float(np.nanmax(dist[first_in[0]:][det[first_in[0]:]])) if len(first_in) else math.nan
        out["pass"] = out["end"] == "completed" and out["t_to_2cm_s"] <= 5.0 and out["last10_cm"] < 1.0
        out["key"] = out["last10_cm"]
    return out


def permutation_p(a: list[float], b: list[float], n: int = 20000, seed: int = 0) -> float:
    """Two-sided permutation test on the difference of means."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return math.nan
    rng = np.random.default_rng(seed)
    observed = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    hits = 0
    for _ in range(n):
        rng.shuffle(pool)
        hits += abs(pool[:len(a)].mean() - pool[len(a):].mean()) >= observed
    return hits / n


def collect(tags: list[str], log_dir: Path, after: str = "", before: str = "") -> tuple[list[dict], list[str]]:
    results, skipped = [], []
    for side in sorted(log_dir.glob("run_*.json")):
        stamp = side.stem[4:]
        if (after and stamp <= after) or (before and stamp >= before):
            continue
        try:
            tag = json.loads(side.read_text(encoding="utf-8")).get("tag", "")
        except (OSError, ValueError):
            skipped.append(f"{stamp}: sidecar unreadable")
            continue
        if tag not in tags:
            continue
        res = analyze_run(side.with_suffix(".csv"))
        if res is None:
            skipped.append(f"{stamp}: log unreadable or too short")
        else:
            results.append(res)
    return results, skipped


def _f(v: float, spec: str = "6.2f") -> str:
    return format(v, spec) if isinstance(v, (int, float)) and math.isfinite(v) else "   -  "


def report(results: list[dict], tags: list[str]) -> str:
    lines = []
    path_rows = [r for r in results if r["mode"] == "path"]
    bal_rows = [r for r in results if r["mode"] != "path"]
    if path_rows:
        lines.append(f"{'run':<18}{'tag':<30}{'end':>12}{'err cm':>8}{'p95':>7}{'ball r':>7}{'ref r':>7}"
                     f"{'peak Hz':>8}{'share':>6}{'LC':>4}{'servo jit':>10}{'still%':>7}{'drops(dark)':>12}  loss")
        for r in path_rows:
            lines.append(f"{r['stamp']:<18}{r['tag'][:29]:<30}{r['end']:>12}{_f(r['error_cm'], '8.2f')}{_f(r['p95_cm'], '7.2f')}"
                         f"{_f(r['ball_r_cm'], '7.2f')}{_f(r['ref_r_cm'], '7.2f')}{_f(r['peak_hz'], '8.2f')}"
                         f"{_f(100 * r['peak_share'], '6.0f')}{'YES' if r['limit_cycle'] else '':>4}{_f(r['servo_jitter'], '10.1f')}"
                         f"{_f(r['still_pct'], '7.0f')}{r['drops']:>7}({r['drops_dark']})   {r['loss']}")
    if bal_rows:
        lines.append(f"{'run':<18}{'tag':<30}{'end':>12}{'start (cm)':>15}{'t->2cm':>7}{'last10':>7}{'overshoot':>10}"
                     f"{'servo jit':>10}{'still%':>7}{'pass':>6}  loss")
        for r in bal_rows:
            sx, sy = r["start_cm"]
            lines.append(f"{r['stamp']:<18}{r['tag'][:29]:<30}{r['end']:>12}{f'({sx:+.1f},{sy:+.1f})':>15}"
                         f"{_f(r['t_to_2cm_s'], '7.1f')}{_f(r['last10_cm'], '7.2f')}{_f(r['overshoot_cm'], '10.1f')}"
                         f"{_f(r['servo_jitter'], '10.1f')}{_f(r['still_pct'], '7.0f')}{'PASS' if r['pass'] else '':>6}  {r['loss']}")
    lines.append("")
    lines.append(f"{'tag':<32}{'n':>3}{'completed':>11}{'key mean':>10}{'key sd':>8}{'LC/pass':>9}{'servo jit':>10}{'p vs first':>11}")
    by_tag = {tag: [r for r in results if r["tag"] == tag] for tag in tags}
    ref_keys = [r["key"] for r in by_tag.get(tags[0], [])]
    for tag in tags:
        rs = by_tag[tag]
        if not rs:
            lines.append(f"{tag:<32}  0  (no runs)")
            continue
        keys = np.array([r["key"] for r in rs], float)
        flag = sum(bool(r.get("limit_cycle")) for r in rs) if rs[0]["mode"] == "path" else sum(bool(r.get("pass")) for r in rs)
        p = "" if tag == tags[0] else _f(permutation_p([r["key"] for r in rs], ref_keys), "11.2f")
        lines.append(f"{tag:<32}{len(rs):3d}{sum(r['end'] == 'completed' for r in rs):>7d}/{len(rs):<3d}"
                     f"{_f(np.nanmean(keys), '10.2f')}{_f(np.nanstd(keys), '8.2f')}{flag:>9d}"
                     f"{_f(np.nanmean([r['servo_jitter'] for r in rs]), '10.1f')}{p:>11}")
    lines.append("key = path: mean tracking error (cm); balance: mean distance from the centre over the last 10 s (cm). "
                 "LC/pass = limit-cycle runs (path) or PASS runs (balance).")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tags", nargs="+")
    parser.add_argument("--after", default="", help="only runs whose stamp is later than this (e.g. 20260928T092615Z)")
    parser.add_argument("--before", default="", help="only runs whose stamp is earlier than this")
    parser.add_argument("--log-dir", type=Path, default=RM.LOG_DIR)
    args = parser.parse_args()
    results, skipped = collect(args.tags, args.log_dir, args.after, args.before)
    print(report(results, args.tags))
    for note in skipped:
        print("skipped", note)


if __name__ == "__main__":
    main()
