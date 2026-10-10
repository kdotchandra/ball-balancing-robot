"""Measure the loop delay of the real platform: tilt the plate sinusoidally, watch the ball.

The stability analysis (stability.py) needs ONE number it cannot get from the design model: the pure
delay between "the ball moved" and "the plate reacted" (camera latency + image processing + servo
command period). This test measures it directly, in open loop, so that the report does not have to
rely on an estimate.

How: the plate is tilted with theta(t) = A*sin(2*pi*f*t) for several cycles while the ball is
tracked. For a ball rolling on a plate the position response is

    P(jw) = -(g_eff / w^2) * 1/(1 + j*w*TA) * exp(-j*w*d) * A          (w = 2*pi*f)

so the phase of the measured position relative to the command gives d at each frequency, and the
amplitude gives g_eff at each frequency. Several frequencies are used because a pure delay shows up
as a phase that grows LINEARLY with frequency (anything else is a modelling error, and the fit
residual says so).

KNOWN LIMITATION (found on the real plate, 24 Sep): this is an OPEN-LOOP test of 10-13 s per frequency. The plate is
never perfectly level, so the ball feels a constant extra acceleration and drifts off (it rolled 15 cm and left the
plate in the 0.4 and 0.6 Hz runs), which buries the small oscillation being measured (SNR 0.1-0.6 instead of >= 3).
The synthetic check that validated the maths assumed a level plate, which is why it looked fine. The real loop
delay needs a closed-loop version (inject the sinusoid on top of the running controller); until that exists
`--enable` is refused unless --force is given, and the result should not be used in the report.

Without --enable nothing is sent to the servos (dry run: only ball tracking is checked). With
--enable --confirm the servos move: keep the ball near the middle of the plate and keep the
emergency power cutoff ready. Every run is logged to logs/freq_*.csv.

    # dry run (no servo motion), check that the ball is tracked
    python test_freq_response.py --axis x

    # real run, x axis, three frequencies, prompts before each
    python test_freq_response.py --axis x --enable --confirm

    # analyse logs that already exist
    python test_freq_response.py --analyze logs/freq_x_0.5Hz_*.csv logs/freq_x_0.8Hz_*.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np

import stability as ST
from params import TA

SETTLE_S = 1.5
BASELINE_S = 1.0
AFTER_S = 1.5
CYCLES = 5
MOVE_MS = 50                  # same servo transition time as the balance runs
DEFAULT_FREQS = (0.4, 0.6, 0.9)
MAX_TILT_DEG = 3.0
LOG_FIELDS = ["t_s", "phase", "tracked", "x_cm", "y_cm", "tilt_cmd_deg", "servo1", "servo2", "servo3"]


def servo_targets(mapping, neutral, lo, hi, axis: str, tilt_deg: float) -> np.ndarray:
    """Same conversion as the balance loop (identical to test_tilt_direction.py)."""
    tilt = np.zeros(2)
    tilt[0 if axis == "x" else 1] = math.radians(tilt_deg)
    proposed = neutral + np.rad2deg(mapping @ tilt) * (1000.0 / 240.0)
    return np.clip(np.rint(proposed), lo, hi).astype(int)


def run_one(tracker, bus, neutral, mapping, lo, hi, axis: str, freq_hz: float, amp_deg: float, log_dir: Path):
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log_path = log_dir / f"freq_{axis}_{freq_hz:g}Hz_{amp_deg:g}deg_{stamp}.csv"
    rows: list[dict] = []
    state = {"servo": neutral.copy(), "tilt": 0.0}

    def send(pos, tilt_deg):
        state["servo"], state["tilt"] = pos, tilt_deg
        if bus is not None:
            for servo_id, p in enumerate(pos, start=1):
                bus.move(servo_id, int(p), MOVE_MS)

    with open(log_path, "w", newline="", encoding="utf-8", buffering=1) as handle:
        writer = csv.writer(handle)
        writer.writerow(LOG_FIELDS)
        t0 = time.perf_counter()

        def watch(phase: str, duration: float, drive: bool = False):
            t_phase = time.perf_counter()
            while time.perf_counter() - t_phase < duration:
                ok, x_m, y_m, *_ = tracker.read()
                now = time.perf_counter() - t0
                if drive:
                    tilt = amp_deg * math.sin(2.0 * math.pi * freq_hz * (time.perf_counter() - t_phase))
                    send(servo_targets(mapping, neutral, lo, hi, axis, tilt), tilt)
                row = {"t_s": now, "phase": phase, "tracked": bool(ok and x_m is not None),
                       "x_cm": x_m * 100.0 if ok and x_m is not None else float("nan"),
                       "y_cm": y_m * 100.0 if ok and y_m is not None else float("nan"),
                       "tilt_cmd_deg": state["tilt"]}
                rows.append(row)
                writer.writerow([f"{now:.4f}", phase, int(row["tracked"]), f"{row['x_cm']:.3f}", f"{row['y_cm']:.3f}",
                                 f"{state['tilt']:.4f}", *(int(v) for v in state["servo"])])

        send(neutral, 0.0)
        watch("settle", SETTLE_S)
        watch("baseline", BASELINE_S)
        if sum(1 for r in rows if r["phase"] == "baseline" and r["tracked"]) < 15:
            return {"ok": False, "detail": "ball not tracked during the baseline; the plate was NOT tilted"}, log_path
        watch("drive", CYCLES / freq_hz, drive=True)
        send(neutral, 0.0)
        watch("after", AFTER_S)
    return analyze_rows(rows, axis, freq_hz), log_path


def analyze_rows(rows: list[dict], axis: str, freq_hz: float) -> dict:
    """Fit position = c0 + c1*t + a*cos(wt) + b*sin(wt) over the drive phase (first cycle dropped)."""
    drive = [r for r in rows if r["phase"] == "drive" and r["tracked"]]
    if len(drive) < 30:
        return {"ok": False, "detail": f"only {len(drive)} tracked frames while driving"}
    t_start = min(r["t_s"] for r in rows if r["phase"] == "drive")
    keep = [r for r in drive if r["t_s"] - t_start >= 1.0 / freq_hz]           # skip the first cycle (start-up transient)
    if len(keep) < 20:
        return {"ok": False, "detail": "too few frames after the first cycle"}
    t = np.array([r["t_s"] - t_start for r in keep])
    y = np.array([r[f"{axis}_cm"] for r in keep]) * 10.0                        # mm
    u = np.array([r["tilt_cmd_deg"] for r in keep])
    w = 2.0 * math.pi * freq_hz
    # the ball direction convention: +tilt_x -> +x ; +tilt_y -> -y (see test_tilt_direction.py)
    sign = 1.0 if axis == "x" else -1.0
    basis = np.stack([np.ones_like(t), t, np.cos(w * t), np.sin(w * t)], axis=1)
    coef, *_ = np.linalg.lstsq(basis, sign * y, rcond=None)
    ucoef, *_ = np.linalg.lstsq(basis, u, rcond=None)                            # the command as actually sent
    resid = sign * y - basis @ coef
    P = complex(coef[3], coef[2])          # response  = Im{P e^{jwt}}
    U = complex(ucoef[3], ucoef[2])        # command   = Im{U e^{jwt}}
    if abs(U) < 1e-9:
        return {"ok": False, "detail": "no commanded tilt found"}
    # P is in mm, U in degrees: convert to metres per radian of tilt
    H = (P * 1e-3) / (U * math.pi / 180.0)
    g_est = abs(H) * w * w * abs(1.0 + 1j * w * TA)
    phase = math.atan2((-H).imag, (-H).real)                                    # angle of -H (ideal plant: 0 rad)
    delay = -(phase + math.atan(w * TA)) / w
    snr = abs(P) / max(float(np.std(resid)), 1e-9)
    return {"ok": True, "freq_hz": freq_hz, "amp_mm": abs(P), "g_eff": g_est, "delay_s": delay,
            "snr": snr, "n": len(keep), "resid_mm": float(np.std(resid))}


def analyze_file(path: Path) -> dict:
    parts = path.stem.split("_")                    # freq_<axis>_<f>Hz_<A>deg_<stamp>
    axis, freq = parts[1], float(parts[2].replace("Hz", ""))
    with open(path, newline="", encoding="utf-8") as handle:
        rows = [{"t_s": float(r["t_s"]), "phase": r["phase"], "tracked": r["tracked"] == "1",
                 "x_cm": float(r["x_cm"]), "y_cm": float(r["y_cm"]), "tilt_cmd_deg": float(r["tilt_cmd_deg"])}
                for r in csv.DictReader(handle)]
    return analyze_rows(rows, axis, freq)


def summarize(results: list[dict]) -> dict:
    good = [r for r in results if r.get("ok") and r["snr"] >= 3.0]
    if not good:
        return {"ok": False}
    delays = np.array([r["delay_s"] for r in good])
    return {"ok": True, "delay_s": float(np.median(delays)), "delay_min": float(delays.min()),
            "delay_max": float(delays.max()), "g_eff": float(np.median([r["g_eff"] for r in good])), "n": len(good)}


def print_results(results: list[tuple[str, dict]]) -> None:
    print("\n================ RESULT ================")
    print(f"{'run':<28}{'freq Hz':>8}{'swing mm':>10}{'SNR':>6}{'g_eff':>8}{'delay s':>9}")
    for name, r in results:
        if not r.get("ok"):
            print(f"{name:<28}  -- {r.get('detail', 'no result')}")
            continue
        print(f"{name:<28}{r['freq_hz']:>8g}{r['amp_mm']:>10.1f}{r['snr']:>6.1f}{r['g_eff']:>8.2f}{r['delay_s']:>9.3f}")
    s = summarize([r for _n, r in results])
    if not s["ok"]:
        print("\nNo run had enough signal (SNR >= 3). Try a larger --amp-deg or check the ball tracking.")
        return
    print(f"\nloop delay (median of {s['n']} runs): {s['delay_s']:.3f} s   (range {s['delay_min']:.3f} - {s['delay_max']:.3f} s)")
    print(f"plant gain g_eff (median): {s['g_eff']:.2f} m/s^2/rad   (design model {ST.G_EFF:.3f})")
    print("A consistent delay across frequencies means the model fits; a delay that changes strongly with "
          "frequency means something other than a pure delay is missing.")
    print(f"Compare with the design estimate {ST.DEFAULT_DELAY_S:g} s (stability.DEFAULT_DELAY_S).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure loop delay and plant gain with a sinusoidal tilt")
    parser.add_argument("--port", default="/dev/serial0")
    parser.add_argument("--axis", choices=("x", "y"), default="x")
    parser.add_argument("--freqs", default=",".join(str(f) for f in DEFAULT_FREQS), help="comma list of frequencies in Hz")
    parser.add_argument("--amp-deg", type=float, default=2.0, help="tilt amplitude in degrees (default 2)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--enable", action="store_true", help="really move the servos (otherwise dry run)")
    parser.add_argument("--analyze", nargs="+", type=Path, help="analyse existing freq_*.csv logs and exit")
    parser.add_argument("--force", action="store_true", help="allow --enable despite the open-loop drift problem (see the docstring)")
    args = parser.parse_args()

    if args.analyze:
        print_results([(p.name, analyze_file(p)) for p in args.analyze])
        return
    freqs = [float(v) for v in args.freqs.split(",") if v.strip()]
    if not 0.5 <= args.amp_deg <= MAX_TILT_DEG:
        raise SystemExit(f"--amp-deg must be between 0.5 and {MAX_TILT_DEG}")
    if any(not 0.2 <= f <= 1.5 for f in freqs):
        raise SystemExit("frequencies must be between 0.2 and 1.5 Hz")

    from ball_tracker import BallTracker
    from servo_calibration.config import load_config, require_values

    limits = load_config("servo_limits.json")
    geometry = load_config("servo_geometry.json")
    mapping_cfg = load_config("servo_mapping.json")
    lo = np.array(require_values(limits, "mechanical_min"), dtype=float)
    hi = np.array(require_values(limits, "mechanical_max"), dtype=float)
    neutral = np.array(require_values(geometry, "neutral_position"), dtype=float).astype(int)
    mapping = np.array([mapping_cfg["axis_mapping"]["tilt_x"], mapping_cfg["axis_mapping"]["tilt_y"]], dtype=float).T
    if args.enable and not args.force:
        raise SystemExit("Refusing --enable: this open-loop test lets the ball drift off the plate (see the docstring). "
                         "Use --force only if you accept that.")
    move = args.enable and args.confirm
    print(f"axis {args.axis}, frequencies {freqs} Hz, amplitude {args.amp_deg:g} deg, {CYCLES} cycles each "
          f"({sum(CYCLES / f for f in freqs):.0f} s of tilting in total)")
    print("REAL RUN: servos WILL move." if move else "DRY RUN: no servo command will be sent.")

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tracker = BallTracker(camera_index=args.camera, mirror=True, detector_mode="old")
    bus = None
    results: list[tuple[str, dict]] = []
    try:
        if move:
            from servo_calibration.hx35h import HX35HBus
            bus = HX35HBus(args.port)
        for k, freq in enumerate(freqs):
            if move:
                input(f"\n[{k + 1}/{len(freqs)}] {freq:g} Hz: ball near the plate centre? cutoff ready? Enter to start, Ctrl+C to abort: ")
            else:
                print(f"\n[{k + 1}/{len(freqs)}] {freq:g} Hz")
            result, path = run_one(tracker, bus, neutral, mapping, lo, hi, args.axis, freq, args.amp_deg, log_dir)
            results.append((path.name, result))
            print("  ->", "ok" if result.get("ok") else result.get("detail"), f"  log: {path.name}")
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
    print_results(results)


if __name__ == "__main__":
    main()
