from __future__ import annotations

import argparse
import json
import csv
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from ball_tracker import BallTracker, probe_cameras
from lqr_controller import LQRController
from params import HOLD_LAST_TIMEOUT_S, K_AXIS, TA, VEL_LIMIT_MPS
from params import SERVO_MAX, SERVO_MIN, SERVO_NEUTRAL
from actuator import ServoGateway, ServoSafety
from tilt_limits import compute_platform_tilt_limits
import experiment_profile as PROF
from path_tuning import DeadZoneLead, select_gain, select_trim_ki
from trajectory import PHASE_NONE, PathReference
from velocity_filter import AlphaBeta
from dither import SquareDither
from friction_comp import FrictionComp
from video_recorder import VideoRecorder


PRINT_INTERVAL_S = 0.25
TERMINAL_LINE_WIDTH = 220
LOST_NEUTRAL_TIMEOUT_S = 0.30
# Bias trim: slow integrator that learns how far the plate is from level at zero tilt.
TRIM_LIMIT_RAD = math.radians(6.0)
TRIM_ACTIVE_RADIUS_M = 0.10   # only learn while the ball is this close to the reference


def deg(rad: float) -> float:
    return math.degrees(rad)


def clip_velocity(v: float) -> float:
    return float(np.clip(v, -VEL_LIMIT_MPS, VEL_LIMIT_MPS))




def print_status_line(line: str) -> None:
    # Overwrite one terminal line to keep output readable during realtime runs.
    padded = line.ljust(TERMINAL_LINE_WIDTH)
    print(f"\r{padded}", end="", flush=True)


def load_servo_safety() -> ServoSafety:
    config_dir = Path(__file__).resolve().parent / "servo_calibration" / "config"
    limits = json.loads((config_dir / "servo_limits.json").read_text(encoding="utf-8"))
    geometry = json.loads((config_dir / "servo_geometry.json").read_text(encoding="utf-8"))
    required = (limits.get("mechanical_min"), limits.get("mechanical_max"), geometry.get("neutral_position"))
    if any(not isinstance(values, list) or len(values) != 3 or any(value is None for value in values) for values in required):
        raise RuntimeError("Complete floor, mechanical limits, and neutral calibration before integrated run")
    return ServoSafety(
        neutral=np.array(geometry["neutral_position"], dtype=int),
        minimum=np.array(limits["mechanical_min"], dtype=int),
        maximum=np.array(limits["mechanical_max"], dtype=int),
    )


def load_servo_mapping() -> np.ndarray:
    config_dir = Path(__file__).resolve().parent / "servo_calibration" / "config"
    mapping = json.loads((config_dir / "servo_mapping.json").read_text(encoding="utf-8"))
    if mapping.get("status") != "mapping_measured":
        raise RuntimeError("Complete servo mapping calibration before integrated run")
    axis_mapping = mapping.get("axis_mapping")
    if not isinstance(axis_mapping, dict):
        raise RuntimeError("servo_mapping.json is missing axis_mapping")
    try:
        matrix = np.array([axis_mapping["tilt_x"], axis_mapping["tilt_y"]], dtype=float).T
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("servo_mapping.json contains invalid axis_mapping") from exc
    if matrix.shape != (3, 2) or not np.isfinite(matrix).all():
        raise RuntimeError("servo_mapping.json axis_mapping must contain 3 finite values per axis")
    return matrix


def load_balance_reference() -> tuple[float, float, float]:
    path = Path(__file__).resolve().parent / "servo_calibration" / "config" / "control_config.json"
    if not path.exists():
        return 0.0, 0.0, 4.0
    config = json.loads(path.read_text(encoding="utf-8"))
    reference = config.get("balance_reference_m", [0.0, 0.0])
    if not isinstance(reference, list) or len(reference) != 2:
        raise RuntimeError("control_config.json balance_reference_m must contain [x_m, y_m]")
    return float(reference[0]), float(reference[1]), float(config.get("reference_radius_cm", 4.0))


LOG_FIELDS = [
    "t_s", "dt_s", "mode", "detected", "control_valid", "confidence",
    "x_m", "y_m", "x_ref_m", "y_ref_m", "x_dot_mps", "y_dot_mps",
    "theta_x_cmd_deg", "theta_y_cmd_deg", "theta_x_actual_deg", "theta_y_actual_deg",
    "theta_x_actuate_deg", "theta_y_actuate_deg", "trim_x_deg", "trim_y_deg",
    "servo1", "servo2", "servo3", "missed_frames", "lost_neutralized",
    "x_ref_dot_mps", "y_ref_dot_mps", "path_phase", "xi_x_ms", "xi_y_ms", "frame_age_s",
    "servo1_fb", "servo2_fb", "servo3_fb", "fc_x_deg", "fc_y_deg",
]


def open_run_log() -> tuple[object, object, Path]:
    """Open a fresh per-run CSV log under logs/, one row per control-loop iteration."""
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8", buffering=1)
    writer = csv.writer(log_file)
    writer.writerow(LOG_FIELDS)
    return log_file, writer, log_path


def write_run_meta(log_path: Path | None, meta: dict) -> None:
    """Sidecar next to the CSV (run_<stamp>.json): what this run was, so nothing has to be remembered."""
    if log_path is None:
        return
    try:
        log_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        print(f"[Startup] could not write the run sidecar: {exc}")




def draw_overlay(
    frame: np.ndarray,
    mode: int,
    detected: bool,
    x_m: float,
    y_m: float,
    theta_x_cmd: float,
    theta_y_cmd: float,
    x_ref: float,
    y_ref: float,
    pixel_per_cm: float,
    balance_x_ref: float,
    balance_y_ref: float,
    reference_radius_cm: float,
) -> None:
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.drawMarker(frame, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 16, 1)
    ref_px = int(round(cx + balance_x_ref * pixel_per_cm * 100.0))
    ref_py = int(round(cy - balance_y_ref * pixel_per_cm * 100.0))
    ref_radius_px = max(5, int(round(reference_radius_cm * pixel_per_cm)))
    cv2.circle(frame, (ref_px, ref_py), ref_radius_px, (255, 0, 255), 3)
    cv2.drawMarker(frame, (ref_px, ref_py), (255, 0, 255), cv2.MARKER_CROSS, 24, 2)
    cv2.putText(frame, "BALANCE REF", (ref_px + 12, ref_py - ref_radius_px - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    lines = [
        f"Mode: {'BALANCE' if mode == 1 else 'TRACKING'} (press M to toggle)",
        f"Detected: {'YES' if detected else 'NO'}",
        f"Ball (mm): x={x_m * 1000:+6.1f}, y={y_m * 1000:+6.1f}",
        f"Tilt cmd (deg): tx={deg(theta_x_cmd):+.2f}, ty={deg(theta_y_cmd):+.2f}",
    ]
    if mode == 2:
        lines.append(f"Ref (mm): x_ref={x_ref * 1000:+6.1f}, y_ref={y_ref * 1000:+6.1f}")

    y0 = 24
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (10, y0 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def draw_ball_marker(frame: np.ndarray, det_px: tuple[float, float, float, float] | None) -> None:
    if det_px is None:
        return

    cx, cy, radius = det_px[:3]
    cx_i = int(round(cx))
    cy_i = int(round(cy))
    radius_i = max(1, int(round(radius)))

    cv2.circle(frame, (cx_i, cy_i), radius_i, (0, 255, 0), 2)
    cv2.circle(frame, (cx_i, cy_i), 4, (0, 0, 255), -1)
    cv2.line(frame, (cx_i - 18, cy_i), (cx_i + 18, cy_i), (0, 0, 255), 2)
    cv2.line(frame, (cx_i, cy_i - 18), (cx_i, cy_i + 18), (0, 0, 255), 2)
    label = f"({cx_i}, {cy_i}) px r={radius_i}"
    label_pos = (cx_i + 12, cy_i - 12)
    cv2.putText(
        frame,
        label,
        label_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def run(
    camera_index: int,
    mirror: bool,
    headless: bool = False,
    tilt_limit_deg: float | None = PROF.TILT_LIMIT_DEG,
    enable_log: bool = True,
    max_tilt_rate_deg_s: float = PROF.MAX_TILT_RATE_DEG_S,
    trim_ki: float = PROF.TRIM_KI,
    trim_init_deg: tuple[float, float] = (0.0, 0.0),
    detector: str = "old",
    save_frames: int = PROF.SAVE_FRAMES,
    k_scale: float = PROF.K_SCALE,
    kv_scale: float = PROF.KV_SCALE,
    k_full: tuple[float, float, float] | None = None,
    path_k_full: tuple[float, float, float] | None = None,
    ta_s: float = TA,
    vel_ab: tuple[float, float] | None = None,
    dither: tuple[float, float] | None = None,
    friction_comp: tuple[float, float] | None = None,
    trim_radius_m: float = TRIM_ACTIVE_RADIUS_M,
    profile_name: str = "",
    path: str = "none",
    path_period_s: float = PROF.PATH_PERIOD_S,
    path_hold_s: float = PROF.PATH_HOLD_S,
    path_ramp_s: float = PROF.PATH_RAMP_S,
    duration_s: float = 0.0,
    path_trim_ki: float | None = None,
    dz_lead_deg: float = 0.0,
    path_size_cm: float | None = None,
    path_ki: float = PROF.PATH_KI,
    tag: str = "",
    record_video: bool = False,
) -> None:
    if tilt_limit_deg is not None and tilt_limit_deg <= 0:
        raise ValueError("tilt_limit_deg must be positive")

    tracker: BallTracker | None = None
    actuator: ServoGateway | None = None
    log_file = None
    log_writer = None
    log_path = None
    recorder: VideoRecorder | None = None
    reference_radius_cm = 4.0
    end_reason = "interrupted"     # replaced when the run ends by itself; stays if Ctrl+C / q / an error
    run_started = time.time()
    meta: dict = {}
    try:
        if enable_log:
            log_file, log_writer, log_path = open_run_log()
            print(f"[Startup] Logging every control-loop iteration to {log_path}")

        safety = load_servo_safety()
        servo_mapping = load_servo_mapping()
        balance_x_ref, balance_y_ref, reference_radius_cm = load_balance_reference()

        if tilt_limit_deg is not None:
            override_rad = np.deg2rad(tilt_limit_deg)
            tilt_x_min_rad, tilt_x_max_rad = -override_rad, override_rad
            tilt_y_min_rad, tilt_y_max_rad = -override_rad, override_rad
            source = "profile default" if abs(tilt_limit_deg - PROF.TILT_LIMIT_DEG) < 1e-9 else "OVERRIDDEN via --tilt-limit-deg"
            print(f"[Startup] Tilt limit: +/-{tilt_limit_deg:.2f} deg (symmetric, both axes; {source})")
        else:
            tilt_limits_by_axis = compute_platform_tilt_limits(safety, servo_mapping)
            tilt_x_min_rad, tilt_x_max_rad = tilt_limits_by_axis["tilt_x"]
            tilt_y_min_rad, tilt_y_max_rad = tilt_limits_by_axis["tilt_y"]
            print(
                "[Startup] Calibration-derived tilt limits "
                f"tilt_x=({deg(tilt_x_min_rad):+.2f},{deg(tilt_x_max_rad):+.2f})deg "
                f"tilt_y=({deg(tilt_y_min_rad):+.2f},{deg(tilt_y_max_rad):+.2f})deg "
                "[servo_limits.json + servo_geometry.json + servo_mapping.json]"
            )
        lqr_internal_limit_rad = max(tilt_x_max_rad, -tilt_x_min_rad, tilt_y_max_rad, -tilt_y_min_rad)
        print(f"[Startup] Max tilt slew rate: {max_tilt_rate_deg_s:.1f} deg/s")
        # --k-full sets all three LQR gains directly (e.g. a K designed on the measured model); otherwise
        # the notebook gain K_AXIS is scaled, which ties the actuator-state gain to the position scale.
        k_eff = np.array(k_full, dtype=float) if k_full is not None else K_AXIS * np.array([k_scale, k_scale * kv_scale, k_scale])
        gain_source = "--k-full" if k_full is not None else f"k-scale {k_scale:.3f}, kv-scale {kv_scale:.2f}"
        print(f"[Startup] Gains: {gain_source} -> "
              f"[position {k_eff[0]:.3f} rad/m, velocity {k_eff[1]:.3f} rad/(m/s), actuator-state {k_eff[2]:.3f}]")
        k_path = np.array(path_k_full, dtype=float) if path_k_full is not None else None
        if k_path is not None and path == "none":
            print("[Startup] WARNING: --path-k-full only acts while a path is running; ignored because --path was not given")
            k_path = None
        if k_path is not None:
            print(f"[Startup] Gain scheduling: ramp and path phases use --path-k-full -> [position {k_path[0]:.3f} rad/m, "
                  f"velocity {k_path[1]:.3f} rad/(m/s), actuator-state {k_path[2]:.3f}]; hold phase keeps the gain above")
        vel_filters = (AlphaBeta(*vel_ab), AlphaBeta(*vel_ab)) if vel_ab is not None else None
        print(f"[Startup] Ball velocity: "
              + (f"alpha-beta filter alpha={vel_ab[0]:g}, beta={vel_ab[1]:g} (--vel-ab)" if vel_ab is not None
                 else "raw one-frame difference"))
        plate_dither = SquareDither(*dither) if dither is not None else None
        if plate_dither is not None:
            print(f"[Startup] Plate dither: square wave +/-{dither[0]:g} deg at {dither[1]:g} Hz on each axis "
                  "(x and y a quarter period apart), added after the slew limiter (--dither)")
        stiction = FrictionComp(*friction_comp) if friction_comp is not None else None
        if stiction is not None:
            print(f"[Startup] Friction compensation: +{friction_comp[0]:g} deg towards the reference while the ball is "
                  f"slower than {friction_comp[1]:g} cm/s (fading to 0 at that speed), off within 0.5 cm (--friction-comp)")
            if vel_ab is None:
                print("[Startup] WARNING: --friction-comp without --vel-ab: the raw velocity noise (~2 cm/s) will switch it randomly")
        print(f"[Startup] Plate-tilt estimator time constant Ta = {ta_s:.3f} s"
              f"{' (params.TA, datasheet)' if abs(ta_s - TA) < 1e-9 else ' (--ta)'}")
        print(
            f"[Startup] Bias trim: ki={trim_ki:.3f} rad/(m*s), limit +/-{math.degrees(TRIM_LIMIT_RAD):.0f} deg"
            if trim_ki > 0.0
            else "[Startup] Bias trim: DISABLED (--trim-ki 0)"
        )
        if trim_ki > 0.0:
            print(f"[Startup] Bias trim learns only within {trim_radius_m * 100:g} cm of the reference"
                  f"{' (--trim-radius-cm)' if abs(trim_radius_m - TRIM_ACTIVE_RADIUS_M) > 1e-9 else ''}")
        if any(trim_init_deg):
            print(f"[Startup] Bias trim starts at x {trim_init_deg[0]:+.2f}, y {trim_init_deg[1]:+.2f} deg "
                  "(--trim-init-deg; also the value it returns to after the ball is lost)")

        if not 0.0 <= path_ki <= PROF.PATH_KI_MAX:
            raise ValueError(f"--path-ki must be between 0 and {PROF.PATH_KI_MAX:g} rad/(m*s) "
                             "(linear analysis: beyond this the phase margin falls quickly)")
        if path == "none" and path_ki > 0.0:
            print("[Startup] WARNING: --path-ki only acts while a path is running; ignored because --path was not given")
            path_ki = 0.0
        if path_ki > 0.0 and path_trim_ki is not None:
            print("[Startup] WARNING: --path-trim-ki is ignored while --path-ki (integral state) is on; "
                  "the bias trim is frozen during the path instead")
            path_trim_ki = None
        dead_zone_lead = DeadZoneLead(dz_lead_deg)
        if path == "none" and (path_trim_ki is not None or dead_zone_lead.enabled):
            print("[Startup] WARNING: --path-trim-ki / --dz-lead-deg only act while a path is running; "
                  "ignored because --path was not given")
            path_trim_ki, dead_zone_lead = None, DeadZoneLead(0.0)

        path_ref = None
        if path != "none":
            path_ref = PathReference(path, path_period_s, path_hold_s, path_ramp_s, size_cm=path_size_cm)
            print(f"[Startup] Path mode: {path_ref.describe()}")
            if path_ref.warning:
                print(f"[Startup] WARNING: {path_ref.warning}")
            print("[Startup] The path clock starts when the ball is first tracked; the platform balances at the "
                  "centre for the hold time first.")
            print(f"[Startup] Path integral (LQI) gain: {path_ki:.3f} rad/(m*s)"
                  f"{' (off)' if path_ki == 0.0 else f', clamp +/-{math.degrees(TRIM_LIMIT_RAD):.0f} deg, bias trim frozen during the path'}")
            print(f"[Startup] Path-only tuning (ramp and path phases; hold and balance keep --trim-ki): "
                  f"trim ki {path_trim_ki if path_trim_ki is not None else trim_ki:.3f}"
                  f"{'' if path_trim_ki is not None else ' (same as --trim-ki)'}, "
                  f"dead-zone lead {dead_zone_lead.lead_deg:.2f} deg"
                  f"{' (off)' if not dead_zone_lead.enabled else ''}")
        if duration_s > 0:
            print(f"[Startup] The run ends by itself {duration_s:g} s after the ball is first tracked "
                  "(or 3 s after the ball is lost).")

        tracker = BallTracker(camera_index=camera_index, mirror=mirror, detector_mode=detector, save_frames_every=save_frames)
        controller = LQRController(k_axis=k_eff.copy(), tilt_limit_rad=lqr_internal_limit_rad)
        actuator = ServoGateway(safety, min_confidence=tracker.min_confidence)
        if actuator.enabled:
            # The tracker's background reference is captured over the first frames, so the
            # platform must already be at neutral and still when that starts.
            actuator.neutralize()
            time.sleep(max(0.5, actuator.move_ms / 1000.0 + 0.3))

        overrides = []
        for name, value, default in (("k_scale", k_scale, PROF.K_SCALE), ("kv_scale", kv_scale, PROF.KV_SCALE),
                                     ("trim_ki", trim_ki, PROF.TRIM_KI), ("tilt_limit_deg", tilt_limit_deg, PROF.TILT_LIMIT_DEG),
                                     ("max_tilt_rate_deg_s", max_tilt_rate_deg_s, PROF.MAX_TILT_RATE_DEG_S),
                                     ("command_period_s", round(actuator.command_period, 4), PROF.COMMAND_PERIOD_S),
                                     ("move_ms", actuator.move_ms, PROF.MOVE_MS),
                                     ("path_ki", path_ki, PROF.PATH_KI),
                                     ("trim_init_x_deg", trim_init_deg[0], 0.0), ("trim_init_y_deg", trim_init_deg[1], 0.0),
                                     ("ta_s", ta_s, TA), ("trim_radius_m", trim_radius_m, TRIM_ACTIVE_RADIUS_M)):
            if value is None or abs(float(value) - float(default)) > 1e-9:
                overrides.append(name)
        if k_full is not None:
            overrides.append("k_full")
        if k_path is not None:
            overrides.append("path_k_full")
        if vel_ab is not None:
            overrides.append("vel_ab")
        if plate_dither is not None:
            overrides.append("dither")
        if stiction is not None:
            overrides.append("friction_comp")
        meta = {
            "tag": tag, "mode": "path" if path_ref is not None else "balance", "shape": path if path_ref is not None else "",
            "path_size_cm": (path_ref.size_m * 100.0) if path_ref is not None else None,
            "path_period_s": path_period_s if path_ref is not None else None,
            "gains": [k_scale, kv_scale], "k_full": list(k_full) if k_full is not None else None,
            "k_effective": [round(float(v), 6) for v in k_eff],
            "path_k_full": [float(v) for v in k_path] if k_path is not None else None, "ta_s": ta_s, "vel_ab": list(vel_ab) if vel_ab is not None else None,
            "dither": list(dither) if dither is not None else None,
            "friction_comp": list(friction_comp) if friction_comp is not None else None,
            "trim_radius_m": trim_radius_m, "profile": profile_name or None, "trim_ki": trim_ki, "trim_init_deg": list(trim_init_deg), "path_ki": path_ki, "path_trim_ki": path_trim_ki,
            "dz_lead_deg": dz_lead_deg, "tilt_limit_deg": tilt_limit_deg,
            "command_period_s": actuator.command_period, "move_ms": actuator.move_ms,
            "servo_output": bool(actuator.enabled), "servo_feedback_log": bool(actuator.feedback_log), "duration_s_requested": duration_s,
            "profile_overrides": overrides, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ended_reason": "running", "valid_attempt": None,
        }
        if record_video:
            stamp = log_path.stem.removeprefix("run_") if log_path is not None else time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
            video_path = Path(__file__).resolve().parent / "logs" / f"video_{stamp}.mp4"
            recorder = VideoRecorder(video_path, tracker.pixel_per_cm)
            meta["video"] = video_path.name
            print(f"[Startup] Recording annotated camera video to {video_path}")
        write_run_meta(log_path, meta)

        mode = 2 if path_ref is not None else 1
        start_t = time.perf_counter()
        path_t0 = None   # set when the ball is first tracked; the path and the duration count from there

        x_last = 0.0
        y_last = 0.0
        have_last_position = False
        missed_time_s = 0.0       # time of the frames missed since the last detection
        x_dot = 0.0
        y_dot = 0.0
        last_detect_t = start_t
        lost_neutralized = False

        # The controller's internal estimate remains a first-order model until platform tilt feedback exists.
        theta_x_actual = 0.0
        theta_y_actual = 0.0

        # Rate-limited signal that actually drives the servos, kept separate from theta_actual
        # (which is the LQR's own internal state estimate, not a hardware smoothing filter).
        theta_x_actuate = 0.0
        theta_y_actuate = 0.0

        # Learned offset between 'zero commanded tilt' and a truly level plate. Added at the
        # actuation stage only, so the LQR's own model (theta_actual) still sees a level plate.
        # --trim-init-deg preloads it with the offset measured in this room, so it does not have to
        # be learned again from 0 (about 30 s at the balance trim gain) at every start and after a loss.
        trim_x0 = math.radians(trim_init_deg[0])
        trim_y0 = math.radians(trim_init_deg[1])
        trim_x = trim_x0
        trim_y = trim_y0
        # Integral (LQI) state of the position error, per axis [m*s]: only integrates while a path is
        # running (ramp and path phases), starts from 0 when the ramp starts.
        xi_x = 0.0
        xi_y = 0.0
        previous_phase = PHASE_NONE


        print(f"Starting integrated test ({'hardware output enabled' if actuator.enabled else 'dry-run; no servo output'})")
        print("Controls: Q=quit, M=toggle mode" if not headless else "Headless mode: use Ctrl+C to quit")
        print(f"Active camera index: {tracker.camera_index}")
        print(f"Mirror: {'ON' if mirror else 'OFF'}")
        last_print_t = start_t

        while True:
            detected, x_m, y_m, frame, mask, dt, det_px = tracker.read()
            now = time.perf_counter()
            control_valid = detected and x_m is not None and y_m is not None and det_px is not None

            if control_valid:
                if now - last_detect_t > HOLD_LAST_TIMEOUT_S or not have_last_position:
                    # first detection (or back after a long loss): no previous position to difference against,
                    # otherwise the very first frame would see a huge fake velocity and kick the plate
                    x_dot = 0.0
                    y_dot = 0.0
                    if vel_filters is not None:
                        vel_filters[0].reset(x_m)
                        vel_filters[1].reset(y_m)
                elif vel_filters is not None:
                    dt_since_valid = dt + missed_time_s
                    x_dot = clip_velocity(vel_filters[0].update(x_m, dt_since_valid))
                    y_dot = clip_velocity(vel_filters[1].update(y_m, dt_since_valid))
                else:
                    # x_last is the last position that was actually detected, so the time to divide by is
                    # everything since THEN (missed frames included), not just the last frame interval.
                    # Dividing by one frame after a dropout made the velocity 2-4x too large and kicked the plate.
                    dt_since_valid = dt + missed_time_s
                    x_dot = clip_velocity((x_m - x_last) / max(1e-3, dt_since_valid))
                    y_dot = clip_velocity((y_m - y_last) / max(1e-3, dt_since_valid))
                missed_time_s = 0.0
                x_last = x_m
                y_last = y_m
                have_last_position = True
                last_detect_t = now
                lost_neutralized = False
            else:
                missed_time_s += dt
                lost_time = now - last_detect_t
                if lost_time > LOST_NEUTRAL_TIMEOUT_S:
                    x_dot = 0.0
                    y_dot = 0.0
                    if not lost_neutralized:
                        if actuator.enabled:
                            actuator.neutralize()
                        # Platform is headed back to neutral, the pose the background
                        # reference was captured in; drop the tilt-adapted background.
                        tracker.reset_background()
                        lost_neutralized = True
                    theta_x_actual = 0.0
                    theta_y_actual = 0.0
                    theta_x_actuate = 0.0
                    theta_y_actuate = 0.0
                    trim_x = trim_x0
                    trim_y = trim_y0
                    xi_x = 0.0
                    xi_y = 0.0
                    dead_zone_lead.reset()
                x_m = x_last
                y_m = y_last

            if control_valid and path_t0 is None:
                path_t0 = now
            t_path = (now - path_t0) if path_t0 is not None else 0.0
            x_ref_dot = y_ref_dot = 0.0
            path_phase = PHASE_NONE
            if mode == 1 or path_ref is None:
                x_ref, y_ref = balance_x_ref, balance_y_ref
            else:
                x_ref, y_ref, x_ref_dot, y_ref_dot = path_ref.at(t_path)
                x_ref += balance_x_ref
                y_ref += balance_y_ref
                path_phase = path_ref.phase(t_path)

            lqi_active = path_ki > 0.0 and path_phase in (2, 3)
            if path_phase == 2 and previous_phase < 2:
                xi_x = xi_y = 0.0                     # the ramp starts: fresh integral
            if not lqi_active:
                xi_x = xi_y = 0.0
            previous_phase = path_phase
            k_active = select_gain(k_eff, k_path, path_phase)
            controller.k_axis = k_active
            theta_x_cmd, theta_y_cmd = controller.compute(
                x_m=x_m,
                y_m=y_m,
                x_dot=x_dot,
                y_dot=y_dot,
                theta_x_actual=theta_x_actual,
                theta_y_actual=theta_y_actual,
                x_ref=x_ref,
                y_ref=y_ref,
                x_ref_dot=x_ref_dot,
                y_ref_dot=y_ref_dot,
                xi_x=xi_x,
                xi_y=xi_y,
                k_i=path_ki if lqi_active else 0.0,
            )
            theta_x_cmd = float(np.clip(theta_x_cmd, tilt_x_min_rad, tilt_x_max_rad))
            theta_y_cmd = float(np.clip(theta_y_cmd, tilt_y_min_rad, tilt_y_max_rad))

            # Learn the plate's level offset. Same sign structure as lqr_controller.py: the
            # position-error part of the x command is -k1*e, of the y command +k1*e. Integrating
            # it slowly removes the steady-state offset a P-type state feedback leaves behind.
            # Conditional integration: skip an axis whose command is already at its limit, and
            # only learn while the ball is being tracked near the (possibly moving) reference.
            # A path needs a faster trim than balancing does, so the gain can be switched for the
            # ramp/path phases only; hold and balance keep --trim-ki untouched.
            if lqi_active and control_valid:
                xi_limit = TRIM_LIMIT_RAD / path_ki
                if tilt_x_min_rad < theta_x_cmd < tilt_x_max_rad:
                    xi_x = float(np.clip(xi_x + (x_m - x_ref) * min(dt, 0.1), -xi_limit, xi_limit))
                if tilt_y_min_rad < theta_y_cmd < tilt_y_max_rad:
                    xi_y = float(np.clip(xi_y + (y_m - y_ref) * min(dt, 0.1), -xi_limit, xi_limit))
            active_trim_ki = 0.0 if lqi_active else select_trim_ki(trim_ki, path_trim_ki, path_phase)
            if active_trim_ki > 0.0 and control_valid:
                err_x = x_m - x_ref
                err_y = y_m - y_ref
                if math.hypot(err_x, err_y) < trim_radius_m:
                    trim_step = active_trim_ki * min(dt, 0.1)
                    if tilt_x_min_rad < theta_x_cmd + trim_x < tilt_x_max_rad:
                        trim_x += trim_step * (-float(k_active[0]) * err_x)
                    if tilt_y_min_rad < theta_y_cmd + trim_y < tilt_y_max_rad:
                        trim_y += trim_step * (float(k_active[0]) * err_y)
                    trim_x = float(np.clip(trim_x, -TRIM_LIMIT_RAD, TRIM_LIMIT_RAD))
                    trim_y = float(np.clip(trim_y, -TRIM_LIMIT_RAD, TRIM_LIMIT_RAD))

            # Slew-rate-limit the signal that actually drives the servos, independent of
            # theta_actual (the LQR's own internal state estimate, updated below unchanged).
            # Dead-zone lead (path phases only, 0 by default): push past the linkage's play in the
            # direction the command is heading. Applied here so the tilt limits and the slew limiter
            # below still bound whatever it adds.
            lead_x = dead_zone_lead.offset("x", theta_x_cmd, path_phase)
            lead_y = dead_zone_lead.offset("y", theta_y_cmd, path_phase)
            fc_x = fc_y = 0.0
            if stiction is not None and control_valid:
                fc_x, fc_y = stiction.offsets(x_m - x_ref, y_m - y_ref, x_dot, y_dot)
            target_x = float(np.clip(theta_x_cmd + trim_x + lead_x + fc_x, tilt_x_min_rad, tilt_x_max_rad))
            target_y = float(np.clip(theta_y_cmd + trim_y + lead_y + fc_y, tilt_y_min_rad, tilt_y_max_rad))
            max_step_rad = np.deg2rad(max_tilt_rate_deg_s) * dt
            theta_x_actuate += float(np.clip(target_x - theta_x_actuate, -max_step_rad, max_step_rad))
            theta_y_actuate += float(np.clip(target_y - theta_y_actuate, -max_step_rad, max_step_rad))

            theta_x_actual += (-(theta_x_actual - theta_x_cmd) / ta_s) * dt
            theta_y_actual += (-(theta_y_actual - theta_y_cmd) / ta_s) * dt
            theta_x_actual = float(np.clip(theta_x_actual, tilt_x_min_rad, tilt_x_max_rad))
            theta_y_actual = float(np.clip(theta_y_actual, tilt_y_min_rad, tilt_y_max_rad))

            if control_valid:
                servo_x, servo_y = theta_x_actuate, theta_y_actuate
                if plate_dither is not None:
                    dither_x, dither_y = plate_dither.offsets(now - start_t)
                    servo_x = float(np.clip(servo_x + dither_x, tilt_x_min_rad, tilt_x_max_rad))
                    servo_y = float(np.clip(servo_y + dither_y, tilt_y_min_rad, tilt_y_max_rad))
                q_mapped_rad = servo_mapping @ np.array([servo_x, servo_y], dtype=float)
                servo_positions = actuator.safety.neutral + np.rint(np.rad2deg(q_mapped_rad) * (1000.0 / 240.0)).astype(int)
                safe_positions = actuator.send_positions(servo_positions, float(det_px[3]))
            else:
                theta_x_cmd = 0.0
                theta_y_cmd = 0.0
                safe_positions = actuator.previous.copy()

            if log_writer is not None:
                log_writer.writerow(
                    [
                        f"{now - start_t:.4f}",
                        f"{dt:.4f}",
                        mode,
                        int(detected),
                        int(control_valid),
                        f"{det_px[3]:.3f}" if det_px is not None else "",
                        f"{x_m:.5f}",
                        f"{y_m:.5f}",
                        f"{x_ref:.5f}",
                        f"{y_ref:.5f}",
                        f"{x_dot:.5f}",
                        f"{y_dot:.5f}",
                        f"{deg(theta_x_cmd):.3f}",
                        f"{deg(theta_y_cmd):.3f}",
                        f"{deg(theta_x_actual):.3f}",
                        f"{deg(theta_y_actual):.3f}",
                        f"{deg(theta_x_actuate):.3f}",
                        f"{deg(theta_y_actuate):.3f}",
                        f"{deg(trim_x):.3f}",
                        f"{deg(trim_y):.3f}",
                        int(safe_positions[0]),
                        int(safe_positions[1]),
                        int(safe_positions[2]),
                        tracker.missed_frames,
                        int(lost_neutralized),
                        f"{x_ref_dot:.5f}",
                        f"{y_ref_dot:.5f}",
                        path_phase,
                        f"{xi_x:.5f}",
                        f"{xi_y:.5f}",
                        f"{tracker.last_frame_age_s:.4f}",
                        *(int(v) if v >= 0 else "" for v in actuator.feedback),
                        f"{deg(fc_x):.3f}",
                        f"{deg(fc_y):.3f}",
                    ]
                )

            if recorder is not None and frame is not None and frame.ndim >= 2 and frame.shape[0] > 10:
                recorder.add(frame, {
                    "t": t_path, "tracking": path_phase in (2, 3), "det_px": det_px if control_valid else None,
                    "x_m": x_m, "y_m": y_m, "x_ref": x_ref, "y_ref": y_ref,
                    "tx_deg": deg(theta_x_cmd), "ty_deg": deg(theta_y_cmd),
                })

            if duration_s > 0:
                if path_t0 is not None and t_path >= duration_s:
                    print(f"\n[END] {duration_s:g} s completed")
                    end_reason = "completed"
                    break
                if path_t0 is not None and not control_valid and now - last_detect_t > 3.0:
                    print("\n[END] ball lost for more than 3 s")
                    end_reason = "ball_lost"
                    break
                if path_t0 is None and now - start_t > 30.0:
                    print("\n[END] ball never tracked within 30 s")
                    end_reason = "never_tracked"
                    break

            if now - last_print_t >= PRINT_INTERVAL_S:
                servo_txt = " ".join(f"{int(v):3d}" for v in safe_positions)
                tilt_txt = f"tilt=({deg(theta_x_cmd):+5.1f},{deg(theta_y_cmd):+5.1f}) deg  servo=[{servo_txt}]"
                if not control_valid and now - last_detect_t > LOST_NEUTRAL_TIMEOUT_S:
                    line = "[LOST] Ball unavailable; output neutralized; waiting for reacquisition"
                elif path_phase in (PHASE_NONE, 1):   # balance run, or the hold phase of a path run
                    label = "[HOLD]" if path_phase == 1 else "[BAL] "
                    line = (f"{label} ball=({x_m * 1000:+6.1f},{y_m * 1000:+6.1f}) mm  "
                            f"dist={math.hypot(x_m - x_ref, y_m - y_ref) * 1000:5.1f} mm  {tilt_txt}")
                else:
                    label = {2: "[RAMP]", 3: "[PATH]"}.get(path_phase, "[TRK] ")
                    line = (f"{label} ball=({x_m * 1000:+6.1f},{y_m * 1000:+6.1f}) mm  "
                            f"ref=({x_ref * 1000:+6.1f},{y_ref * 1000:+6.1f}) mm  "
                            f"err={math.hypot(x_m - x_ref, y_m - y_ref) * 1000:5.1f} mm  {tilt_txt}")
                print_status_line(line)
                last_print_t = now

            if not headless:
                if frame is None or frame.ndim != 3 or frame.shape[0] < 220 or frame.shape[1] < 360:
                    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                if mask is None or mask.ndim != 2 or mask.shape[0] < 10 or mask.shape[1] < 10:
                    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

                draw_ball_marker(frame, det_px)
                draw_overlay(
                    frame=frame,
                    mode=mode,
                    detected=detected,
                    x_m=x_m,
                    y_m=y_m,
                    theta_x_cmd=theta_x_cmd,
                    theta_y_cmd=theta_y_cmd,
                    x_ref=x_ref,
                    y_ref=y_ref,
                    pixel_per_cm=tracker.pixel_per_cm,
                    balance_x_ref=balance_x_ref,
                    balance_y_ref=balance_y_ref,
                    reference_radius_cm=reference_radius_cm,
                )

                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(contour) >= 100:
                        cv2.drawContours(mask_bgr, [contour], -1, (0, 0, 255), 2)
                        cv2.drawMarker(
                            mask_bgr,
                            (mask_bgr.shape[1] // 2, mask_bgr.shape[0] // 2),
                            (255, 255, 255),
                            cv2.MARKER_CROSS,
                            12,
                            1,
                        )
                mask_small = cv2.resize(mask_bgr, (320, 180))
                h, w = frame.shape[:2]
                x0 = w - 10 - mask_small.shape[1]
                x1 = w - 10
                y0 = 10
                y1 = 10 + mask_small.shape[0]
                if x0 >= 0 and y1 <= h:
                    frame[y0:y1, x0:x1] = mask_small

                cv2.imshow("Integrated LQR + IK Test", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if key in (ord("m"), ord("M")):
                    print()
                    if path_ref is None:
                        print("No path selected (start with --path circle|ellipse|hexagon); staying in BALANCE")
                    else:
                        mode = 2 if mode == 1 else 1
                        path_t0 = None if mode == 2 else path_t0
                        print(f"Switched mode => {'BALANCE' if mode == 1 else 'TRACKING'}")

    finally:
        print()
        if tracker is not None:
            tracker.release()
        if actuator is not None:
            actuator.close()
        if log_file is not None:
            log_file.close()
        if recorder is not None:
            recorder.close()
        if not headless:
            cv2.destroyAllWindows()
        if meta:
            meta["ended_reason"] = end_reason
            meta["valid_attempt"] = end_reason != "never_tracked"     # never seen = set-up problem, not a result
            meta["duration_s_actual"] = round(time.time() - run_started, 2)
            write_run_meta(log_path, meta)
        if log_path is not None:
            try:
                from run_metrics import print_summary
                print_summary([log_path], reference_radius_cm=reference_radius_cm)
            except Exception as exc:   # the summary must never hide the real error
                print(f"[Metrics] could not summarise the run: {exc}")


def _tilt_limit_arg(text: str) -> float | None:
    """--tilt-limit-deg accepts a number of degrees, or the word 'calibration'."""
    if text.strip().lower() in ("calibration", "cal", "none"):
        return None
    return float(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrated LQR + IK real-time print test")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index. Default: 0",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available camera candidates and exit",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal mirror (default is mirror ON)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without OpenCV GUI, suitable for SSH",
    )
    parser.add_argument(
        "--tilt-limit-deg",
        type=_tilt_limit_arg,
        default=PROF.TILT_LIMIT_DEG,
        help=(
            "Maximum commanded platform tilt in degrees, symmetric on both axes "
            f"(default {PROF.TILT_LIMIT_DEG:g}, from experiment_profile.py). "
            "Use 'calibration' to derive asymmetric per-axis limits from the servo calibration files instead."
        ),
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable per-iteration CSV logging to logs/run_<timestamp>.csv (logging is on by default)",
    )
    parser.add_argument(
        "--max-tilt-rate-deg-s",
        type=float,
        default=PROF.MAX_TILT_RATE_DEG_S,
        help=f"Maximum rate of change of the commanded platform tilt sent to the servos, in deg/s (default: {PROF.MAX_TILT_RATE_DEG_S:g})",
    )
    parser.add_argument(
        "--detector",
        choices=("old", "bgsub"),
        default="old",
        help="Ball detector: 'old' = brightness+Hough (no empty-platform step), 'bgsub' = background subtraction (default: old)",
    )
    parser.add_argument(
        "--save-frames",
        type=int,
        default=PROF.SAVE_FRAMES,
        help=f"Save every Nth processed frame (frame | diff | background) to logs/frames_<time>/ for debugging; 0 = off (default {PROF.SAVE_FRAMES})",
    )
    parser.add_argument("--record-video", action="store_true",
                        help="Record the camera view with ball, reference and trail drawn on it to logs/video_<run stamp>.mp4")
    parser.add_argument(
        "--k-scale",
        type=float,
        default=PROF.K_SCALE,
        help=f"Multiply all LQR gains by this factor. Lower = gentler (default {PROF.K_SCALE:g}: the balance gain set)",
    )
    parser.add_argument(
        "--kv-scale",
        type=float,
        default=PROF.KV_SCALE,
        help=f"Extra multiplier on the velocity gain only (on top of --k-scale) (default {PROF.KV_SCALE:g}: the balance gain set)",
    )
    parser.add_argument(
        "--trim-ki",
        type=float,
        default=PROF.TRIM_KI,
        help=f"Gain of the slow bias-trim integrator in rad/(m*s); 0 disables it (default {PROF.TRIM_KI:g})",
    )
    parser.add_argument(
        "--k-full",
        default=None,
        help="All three LQR gains 'K1,K2,K3' [rad/m, rad/(m/s), -] instead of K_AXIS x --k-scale/--kv-scale, "
             "e.g. a K designed on the measured model (default: off)",
    )
    parser.add_argument(
        "--path-k-full",
        default=None,
        help="Gain scheduling: LQR gains 'K1,K2,K3' used only in the ramp and path phases of a --path run; "
             "the hold phase keeps the main gain (default: off, one gain throughout)",
    )
    parser.add_argument(
        "--vel-ab",
        default=None,
        help="Filter the ball velocity with an alpha-beta filter 'ALPHA,BETA', e.g. 0.5,0.15 "
             "(default: off, raw one-frame difference)",
    )
    parser.add_argument(
        "--dither",
        default=None,
        help="Square-wave plate dither 'AMP_DEG,FREQ_HZ' added after the slew limiter, e.g. 0.4,5 "
             "(default: off)",
    )
    parser.add_argument(
        "--friction-comp",
        default=None,
        help="Stiction compensation 'U_DEG,V0_CMS': extra tilt U towards the reference while the ball is slower "
             "than V0 (fades to 0 at V0), e.g. 0.6,2; use with --vel-ab (default: off)",
    )
    parser.add_argument(
        "--profile",
        default="",
        help="Name of the experiment_profile.PROFILES entry the launcher expanded into flags; recorded in the "
             "run's sidecar only (the flags themselves carry the values)",
    )
    parser.add_argument(
        "--trim-radius-cm",
        type=float,
        default=TRIM_ACTIVE_RADIUS_M * 100.0,
        help=f"The bias trim learns only while the ball is within this distance of the reference "
             f"(default {TRIM_ACTIVE_RADIUS_M * 100:g} cm). A small radius stops it winding up while the ball "
             "is being pulled in from far away",
    )
    parser.add_argument(
        "--ta",
        type=float,
        default=TA,
        help=f"Servo/plate time constant [s] used by the plate-tilt estimator (default {TA:g} from params.TA, datasheet)",
    )
    parser.add_argument(
        "--trim-init-deg",
        default="0,0",
        help="Starting value of the bias trim as 'X,Y' in degrees, e.g. the trim_x_deg/trim_y_deg a run in "
             "this room settled at (default 0,0 = learn from level)",
    )
    parser.add_argument("--path", choices=("none", "circle", "ellipse", "hexagon"), default="none",
                        help="Follow a moving reference path instead of holding the centre (default: none)")
    parser.add_argument("--path-period-s", type=float, default=PROF.PATH_PERIOD_S, help=f"Seconds per lap (default {PROF.PATH_PERIOD_S:g})")
    parser.add_argument("--path-hold-s", type=float, default=PROF.PATH_HOLD_S, help=f"Seconds balancing at the centre before the path starts (default {PROF.PATH_HOLD_S:g})")
    parser.add_argument("--path-ramp-s", type=float, default=PROF.PATH_RAMP_S, help=f"Seconds over which the path grows from the centre (default {PROF.PATH_RAMP_S:g})")
    parser.add_argument("--path-ki", type=float, default=PROF.PATH_KI,
                        help=f"Integral (LQI) gain on the tracking error while the path moves, rad/(m*s), 0 = off, max {PROF.PATH_KI_MAX:g} "
                             f"(default {PROF.PATH_KI:g}). Balance runs and the hold phase are unaffected")
    parser.add_argument("--tag", default="", help="Label stored in the run's sidecar (balance / circle / hexagon ...) so the report can group runs")
    parser.add_argument("--duration-s", type=float, default=0.0,
                        help="End the run this many seconds after the ball is first tracked; 0 = until Ctrl+C")
    parser.add_argument("--path-size-cm", type=float, default=None,
                        help="Override the path size in cm (circle radius / ellipse semi-major / hexagon circumradius). "
                             "Default: the fixed 3 cm. Limited to 7 cm; this is for the size-scaling experiment")
    parser.add_argument("--path-trim-ki", type=float, default=None,
                        help="Bias-trim gain to use while the path is moving (default: same as --trim-ki). "
                             "Balance runs and the hold phase are unaffected")
    parser.add_argument("--dz-lead-deg", type=float, default=0.0,
                        help="Push the tilt command this many degrees past the plate's play, in the direction "
                             "the command is heading, while the path is moving (default 0 = off)")
    args = parser.parse_args()
    if args.path_size_cm is not None and not 1.0 <= args.path_size_cm <= 7.0:
        parser.error("--path-size-cm must be between 1 and 7 (the camera loses the ball beyond about 9 cm)")
    if args.path_trim_ki is not None and not 0.0 <= args.path_trim_ki <= 1.0:
        parser.error("--path-trim-ki must be between 0 and 1")
    if not 0.0 <= args.dz_lead_deg <= 1.0:
        parser.error("--dz-lead-deg must be between 0 and 1 (the simulation lost stability above 1 deg)")
    try:
        args.trim_init_deg = tuple(float(v) for v in args.trim_init_deg.split(","))
    except ValueError:
        args.trim_init_deg = ()
    if len(args.trim_init_deg) != 2 or not all(abs(v) <= math.degrees(TRIM_LIMIT_RAD) for v in args.trim_init_deg):
        parser.error(f"--trim-init-deg must be 'X,Y' with each within +/-{math.degrees(TRIM_LIMIT_RAD):g} deg")
    if args.k_full is not None:
        try:
            args.k_full = tuple(float(v) for v in args.k_full.split(","))
        except ValueError:
            args.k_full = ()
        if len(args.k_full) != 3 or not all(0.0 <= v <= 6.0 for v in args.k_full):
            parser.error("--k-full must be 'K1,K2,K3' with each between 0 and 6")
        if abs(args.k_scale - PROF.K_SCALE) > 1e-9 or abs(args.kv_scale - PROF.KV_SCALE) > 1e-9:
            parser.error("--k-full replaces --k-scale/--kv-scale; give one or the other")
    if args.path_k_full is not None:
        try:
            args.path_k_full = tuple(float(v) for v in args.path_k_full.split(","))
        except ValueError:
            args.path_k_full = ()
        if len(args.path_k_full) != 3 or not all(0.0 <= v <= 6.0 for v in args.path_k_full):
            parser.error("--path-k-full must be 'K1,K2,K3' with each between 0 and 6")
    if args.vel_ab is not None:
        try:
            args.vel_ab = tuple(float(v) for v in args.vel_ab.split(","))
        except ValueError:
            args.vel_ab = ()
        if len(args.vel_ab) != 2 or not (0.0 < args.vel_ab[0] <= 1.0 and 0.0 < args.vel_ab[1] <= 2.0):
            parser.error("--vel-ab must be 'ALPHA,BETA' with 0 < ALPHA <= 1 and 0 < BETA <= 2")
    if args.dither is not None:
        try:
            args.dither = tuple(float(v) for v in args.dither.split(","))
        except ValueError:
            args.dither = ()
        if len(args.dither) != 2 or not (0.0 < args.dither[0] <= 1.0 and 0.5 <= args.dither[1] <= 15.0):
            parser.error("--dither must be 'AMP_DEG,FREQ_HZ' with 0 < AMP_DEG <= 1 and 0.5 <= FREQ_HZ <= 15")
    if args.friction_comp is not None:
        try:
            args.friction_comp = tuple(float(v) for v in args.friction_comp.split(","))
        except ValueError:
            args.friction_comp = ()
        if len(args.friction_comp) != 2 or not (0.0 < args.friction_comp[0] <= 2.0 and 0.2 <= args.friction_comp[1] <= 10.0):
            parser.error("--friction-comp must be 'U_DEG,V0_CMS' with 0 < U_DEG <= 2 and 0.2 <= V0_CMS <= 10")
    if not 0.5 <= args.trim_radius_cm <= 15.0:
        parser.error("--trim-radius-cm must be between 0.5 and 15")
    if not 0.02 <= args.ta <= 1.0:
        parser.error("--ta must be between 0.02 and 1.0 s")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.list_cameras:
        cams = probe_cameras(max_index=8)
        if not cams:
            print("No camera candidate found")
        else:
            print("Camera candidates:")
            for cam in cams:
                print(
                    f"  index={cam['index']} size={cam['width']}x{cam['height']} "
                    f"fps={cam['fps']:.1f} backend={cam['backend']} score={cam['score']:.2f}"
                )
    else:
        run(
            camera_index=args.camera,
            mirror=not args.no_mirror,
            headless=args.headless,
            tilt_limit_deg=args.tilt_limit_deg,
            enable_log=not args.no_log,
            max_tilt_rate_deg_s=args.max_tilt_rate_deg_s,
            trim_ki=args.trim_ki,
            trim_init_deg=args.trim_init_deg,
            detector=args.detector,
            save_frames=args.save_frames,
            k_scale=args.k_scale,
            kv_scale=args.kv_scale,
            k_full=args.k_full,
            path_k_full=args.path_k_full,
            ta_s=args.ta,
            vel_ab=args.vel_ab,
            dither=args.dither,
            friction_comp=args.friction_comp,
            trim_radius_m=args.trim_radius_cm / 100.0,
            profile_name=args.profile,
            path=args.path,
            path_period_s=args.path_period_s,
            path_hold_s=args.path_hold_s,
            path_ramp_s=args.path_ramp_s,
            duration_s=args.duration_s,
            path_trim_ki=args.path_trim_ki,
            dz_lead_deg=args.dz_lead_deg,
            path_size_cm=args.path_size_cm,
            path_ki=args.path_ki,
            tag=args.tag,
            record_video=args.record_video,
        )
