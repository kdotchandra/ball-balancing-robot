"""Theoretical limits of this ball-and-plate system, derived from the project's own parameters.

Every number here comes from `params.py`, the calibration files, or a measurement made with one of
the test scripts in this folder -- nothing is a hand-picked constant. This is what gives the
evaluation targets in `run_metrics.py` / `metrics_report.py` a source that can be checked.

Three levels are worth telling apart, and the report prints all three:

  1. ideal theory     -- the linear design the LQR gain was computed for: ball as a rolling shell
                         on a massless, instant, perfectly level plate.
  2. hardware-limited -- the same design once the MEASURED loop delay and the MEASURED dead zone
                         are included. This is the level a target should be set at.
  3. measured         -- what the run logs actually show (computed in run_metrics.py).

Run `python theory_limits.py` to print the whole table.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from params import G, K_AXIS, KR, TA
import stability as ST

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALIB_PATH = PROJECT_ROOT / "ping_pong_tracker" / "calib_config.json"

BALL_DIAMETER_MM = 40.0        # params.BALL_DIAMETER, in mm for the report
LOOP_RATE_HZ = 30.0            # camera frame rate; every run log shows 30.0 Hz

# --------------------------------------------------------------------------------------
# Measured constants. Each one names the test that produced it, so the report can cite it
# instead of asserting it. Update these when a measurement is repeated.
# --------------------------------------------------------------------------------------
MEASURED = {
    "plant_gain_x": (4.0, "m/s^2/rad", "test_tilt_direction.py --axis x --sweep 0.3,0.6,1,2"),
    "plant_gain_y": (6.1, "m/s^2/rad", "test_tilt_direction.py --axis y --sweep 0.3,0.6,1,2"),
    "dead_zone_x": (0.5, "deg", "tilt sweep: the ball does not move below this commanded tilt"),
    "dead_zone_y": (0.25, "deg", "tilt sweep: between 0.15 and 0.3 deg on this axis"),
    "dead_zone_effective": (0.8, "deg", "HYPOTHESIS, not used for targets: fitted to the first path runs (gains 0.15/1.8) "
                            "but it did NOT predict the runs at the balance gains -- see dead_zone_floor_mm"),
    "friction_breakaway": (1.46, "deg", "rolling friction 0.15 m/s^2 fitted in the closed-loop simulation"),
    "loop_delay": (ST.DEFAULT_DELAY_S, "s", "ESTIMATE, not a direct measurement: about 3 camera frames plus the servo command "
                   "period. The open-loop tilt tests only bound it: the ball starts to move 0.17-0.23 s after the command, "
                   "with a 250 ms servo ramp included. Expected range "
                   f"{ST.DELAY_RANGE_S[0]:g}-{ST.DELAY_RANGE_S[1]:g} s; test_freq_response.py measures it"),
    "detection_noise": (0.75, "mm", "high-frequency spread of x_m/y_m while the ball is at rest"),

    # ---- measured 2026-09-27/28 with --log-servo-feedback (servo positions read back while balancing) ----
    # These describe the hardware for further tuning (experiment_profile.PROFILES uses them). They are NOT
    # used for the chapter-4 targets above: those keep the values they were set with, so the report's
    # numbers do not move. stability.DEFAULT_DELAY_S (0.13 s) is deliberately left as it was for the same reason.
    "servo_time_constant": (0.11, "s", "servo command -> read-back position lag 125 ms minus ~15 ms age of the "
                            "background read; tags presentation_room_servo_fb*, presentation_room_static_check "
                            "(params.TA = 0.181 s is the datasheet value)"),
    "delay_camera": (0.011, "s", "frame_age_s median in the run logs (sensor read-out to control loop)"),
    "delay_vision": (0.014, "s", "BallTracker detection time per frame, timed on 200 recorded frames"),
    "delay_command_hold": (0.033, "s", "commands go out every 2nd 33 ms frame (40 ms minimum period): half of 67 ms"),
    "delay_servo": (0.110, "s", "same measurement as servo_time_constant"),
    "delay_deadzone_reversal": (0.071, "s", "extra time for a servo to move 2 units after its command reverses "
                                "(166 ms) versus continuing in the same direction (95 ms); 1107 / 824 events"),
    "servo_deadband": (3.0, "units", "p90 |read-back - command| while the command is held >= 0.3 s "
                       "(1 unit ~ 0.1 deg of plate tilt on servo 1)"),
    "plant_gain_rolling": (0.113, "m/s^2/deg", "ball acceleration vs read-back plate tilt while rolling > 3 cm/s, "
                           "joint fit of 383 samples with level offset and friction (model 0.6 g = 0.103)"),
    "static_breakaway_min": (0.5, "deg", "plate tilt the ball stood still at: medians of 26 stuck episodes "
                             "(>= 0.7 s, 6 runs) ran 0.5-2 deg, one reached 3.5 deg"),
    "static_breakaway_max": (2.0, "deg", "see static_breakaway_min"),
    "velocity_noise_raw": (31.0, "mm/s", "raw one-frame-difference velocity (std of the high-frequency part); "
                           "0.4-1.1 deg of plate jitter through the velocity gain"),
    "velocity_noise_alpha_beta": (6.2, "mm/s", "alpha-beta 0.7/0.35 replayed on the same logs, ~20 ms extra lag"),
    "dz_lead_limit_cycle_circle": (0.70, "deg", "dead-zone lead where the circle fell into a 0.74 Hz limit cycle "
                                   "(sweep_dz070_circle); 0-0.55 deg did not"),
    "dz_lead_limit_cycle_hexagon": (0.25, "deg", "lowest dead-zone lead that sent the hexagon into a ~0.6-0.8 Hz "
                                    "limit cycle (1 of 4 runs at 0.25, 1 of 5 at 0.40); none in 6 runs without it"),
}


def measured(name: str) -> float:
    return float(MEASURED[name][0])


def measured_source(name: str) -> str:
    value, unit, source = MEASURED[name]
    return f"{value:g} {unit} ({source})"


# --------------------------------------------------------------------------------------
# Plant
# --------------------------------------------------------------------------------------
def plant_gain() -> tuple[float, str]:
    """Acceleration per radian of plate tilt for a rolling ball, and how it is derived.

    A body rolling without slipping down a slope accelerates at a = g*sin(theta)/(1 + I/(m*r^2)).
    A ping-pong ball is a thin spherical shell, I = (2/3)*m*r^2, so a = (3/5)*g*sin(theta).
    params.KR is exactly that 3/5 factor, so the project's model gain is KR*G.
    """
    shell_factor = 1.0 / (1.0 + 2.0 / 3.0)      # = 3/5, the thin spherical shell
    derivation = (f"a = g*sin(theta)/(1+I/mr^2), thin shell I=(2/3)mr^2 -> factor {shell_factor:.3f} "
                  f"(params.KR = {KR:g}), so g_eff = KR*G = {KR * G:.3f} m/s^2/rad")
    return KR * G, derivation


def gains(k_scale: float, kv_scale: float) -> tuple[float, float, float]:
    """(k1, k2, k3) actually used by the controller for these command-line scales.

    main.py builds them as K_AXIS * [k_scale, k_scale*kv_scale, k_scale].
    """
    return float(K_AXIS[0] * k_scale), float(K_AXIS[1] * k_scale * kv_scale), float(K_AXIS[2] * k_scale)


# --------------------------------------------------------------------------------------
# Level 1: ideal linear theory
# --------------------------------------------------------------------------------------
@dataclass
class ClosedLoop:
    k1: float
    k2: float
    omega_n: float
    zeta: float
    overshoot_pct: float
    settle_2pct_s: float
    settle_5pct_s: float


def closed_loop(k_scale: float, kv_scale: float) -> ClosedLoop:
    """Ideal response of the design model (stability.py): three states, no delay.

    omega_n and zeta describe the dominant complex pole pair; overshoot and settling time come from
    the exact time response after releasing the ball 6 cm from the target.
    """
    model = ST.Model(k_scale, kv_scale)
    poles = ST.poles(model)
    pair = max((p for p in poles if p.imag > 1e-9), key=lambda p: p.imag, default=None)
    if pair is None:                       # overdamped: use the slowest real pole
        slow = max(poles, key=lambda p: p.real)
        omega_n, zeta = abs(slow.real), 1.0
    else:
        omega_n = abs(pair)
        zeta = -pair.real / omega_n
    metrics = ST.step_metrics(model)
    return ClosedLoop(model.K[0], model.K[1], float(omega_n), float(zeta), metrics["overshoot_pct"],
                      metrics["settle_2pct_s"], metrics["settle_2pct_s"] * 3.0 / 4.0)


def tracking_error(k_scale: float, kv_scale: float, amplitude_m: float, period_s: float,
                   delay_s: float = 0.0) -> tuple[float, float, float]:
    """Steady-state error following a sinusoid, from the design model's frequency response.

    The error is driven by the reference ACCELERATION (the controller already uses the reference
    velocity), so |e| = |G_ed(jw)| * A*w^2 with the exact three-state G_ed (stability.py).
    Returns (peak_mm, mean_mm, fraction_of_amplitude). A full-circle path is two such axes.
    """
    peak = ST.tracking_error_amplitude(ST.Model(k_scale, kv_scale), amplitude_m, period_s, delay_s) * 1000.0
    return peak, peak * 2.0 / math.pi, peak / (amplitude_m * 1000.0)


# --------------------------------------------------------------------------------------
# Level 2: the same design limited by measured hardware
# --------------------------------------------------------------------------------------
def phase_margin(k_scale: float, kv_scale: float, delay_s: float | None = None) -> tuple[float, float]:
    """(phase margin in degrees, crossover in rad/s) of the design loop with a pure delay added.

    The delay defaults to the ESTIMATE in MEASURED['loop_delay'] (stability.DEFAULT_DELAY_S); pass
    another value to see how sensitive the margin is to it.
    """
    delay = measured("loop_delay") if delay_s is None else delay_s
    m = ST.margins(ST.Model(k_scale, kv_scale), delay)
    return m["pm_deg"], m["wc"]


def sensing_floor_mm() -> tuple[float, float, str]:
    """(mm per pixel, measured noise in mm, note) -- the best any controller here could hold."""
    pixel_per_cm = float(json.loads(CALIB_PATH.read_text())["pixel_per_cm"])
    mm_per_px = 10.0 / pixel_per_cm
    noise = measured("detection_noise")
    note = (f"{pixel_per_cm:.2f} px/cm -> 1 px = {mm_per_px:.3f} mm; measured noise {noise:g} mm "
            f"= {noise / mm_per_px:.1f} px; one pixel per frame at {LOOP_RATE_HZ:g} Hz "
            f"= {mm_per_px * LOOP_RATE_HZ:.1f} mm/s of velocity noise")
    return mm_per_px, noise, note


def dead_zone_floor_mm(k_scale: float, dead_zone_deg: float | None = None) -> float:
    """HYPOTHESIS (falsified for gain scaling): position error needed to command the break-away tilt.

    This predicted a floor of 18 mm at the path gains (k1 0.785) and 35 mm at the balance gains
    (k1 0.398). The real plate gave the SAME mean tracking error, 22 mm, at both gains, with the
    ball swinging 14-19 mm either way. So the error floor does not scale as theta_dead/k1, and this
    function must not be used to set or explain a target. It is kept only so the report can show the
    prediction next to the measurement.

    Original reasoning:
    Position error needed before the controller commands enough tilt to break the ball loose.

    The plate has backlash and rolling friction: below theta_dead the ball does not move at all.
    The controller only reaches that tilt once the position error is theta_dead/k1, so THAT is the
    error floor whenever the ball has to be moved (path following, and the last part of a settle).
    Holding still is different -- friction helps there, and the trim integrator supplies the bias.
    """
    k1, _, _ = gains(k_scale, 1.0)
    theta = math.radians(measured("dead_zone_effective") if dead_zone_deg is None else dead_zone_deg)
    return theta / k1 * 1000.0


def path_amplitude_prediction_mm(k_scale: float, amplitude_m: float, dead_zone_deg: float | None = None) -> float:
    """How far the ball should actually swing on a path of amplitude A: A - theta_dead/k1.

    The ball only follows the part of the reference that produces more than the break-away tilt.
    """
    return max(0.0, amplitude_m * 1000.0 - dead_zone_floor_mm(k_scale, dead_zone_deg))


def null_tracking_error_mm(amplitude_m: float) -> float:
    """Mean tracking error of a 'controller' that just leaves the ball at the centre.

    On a circle of radius A the reference is always A away from the centre, so the error is exactly
    A. This is the natural yardstick for a tracking target: doing nothing scores A, so a controller
    has to beat it by a stated margin for the tracking to count as real.
    """
    return amplitude_m * 1000.0


def reference_tilt_demand_deg(amplitude_m: float, period_s: float) -> float:
    """Tilt the path itself needs (centripetal a = A*w^2), ignoring any error correction.

    If this is far below the dead zone, the ball cannot glide along the path: it must stick and slip.
    """
    g, _ = plant_gain()
    w = 2.0 * math.pi / period_s
    return math.degrees(amplitude_m * w ** 2 / g)


# --------------------------------------------------------------------------------------
# Targets, each with the source that sets it
# --------------------------------------------------------------------------------------
@dataclass
class Target:
    key: str
    label_th: str
    metric: str
    limit: float
    unit: str
    source_th: str
    theory_value: float | None = None
    theory_note_th: str = ""


def targets(k_scale: float = 0.076, kv_scale: float = 2.8,
            path_k_scale: float = 0.15, path_kv_scale: float = 1.8,
            path_amplitude_m: float = 0.030, path_period_s: float = 20.0) -> list[Target]:
    """The evaluation table, with every limit traced back to theory or to a measurement.

    k_scale/kv_scale are the gains used for the balance runs; path_* are the gains used for the
    path runs, because the theoretical values differ between the two tunings.
    """
    g, _ = plant_gain()
    cl = closed_loop(k_scale, kv_scale)
    _, noise_mm, _ = sensing_floor_mm()
    null_mm = null_tracking_error_mm(path_amplitude_m)
    _, path_mean_mm, _ = tracking_error(path_k_scale, path_kv_scale, path_amplitude_m, path_period_s)
    return [
        Target("plant_gain", "ความถูกต้องของแบบจำลองลูกบอล", "g_eff", 0.0, "m/s^2/rad",
               "ทฤษฎีลูกทรงกลมกลวง a = (3/5)g·sinθ",
               g, f"ทฤษฎี {g:.3f}; ยอมรับได้ถ้าวัดได้ 60-110% ของค่านี้"),
        Target("rms_mm", "รักษาลูกบอลให้อยู่ใกล้จุดศูนย์กลาง", "RMS position error", 10.0, "mm",
               f"= 1/4 ของเส้นผ่านศูนย์กลางลูกบอล ({BALL_DIAMETER_MM:g} mm) และประมาณ "
               f"{10.0 / noise_mm:.0f} เท่าของพื้นความละเอียดกล้อง ({noise_mm:g} mm)",
               noise_mm, f"พื้นจำกัดจากเซนเซอร์ {noise_mm:g} mm"),
        Target("settle_s", "ทำให้ระบบกลับสู่สมดุล", "settling time", 7.0, "s",
               f"= 2 เท่าของทฤษฎีอันดับสอง t_settle(2%) = {cl.settle_2pct_s:.2f} s ที่เกน balance "
               f"(โซนศูนย์กลางรัศมี {BALL_DIAMETER_MM:g} mm = หนึ่งเส้นผ่านศูนย์กลางลูกบอล)",
               cl.settle_2pct_s, f"ทฤษฎี {cl.settle_2pct_s:.2f} s (ζ={cl.zeta:.2f}, ωn={cl.omega_n:.2f} rad/s)"),
        Target("steady_mm", "รักษาตำแหน่งเมื่อระบบนิ่ง", "steady-state error", 20.0, "mm",
               f"= รัศมีลูกบอล ({BALL_DIAMETER_MM / 2:g} mm) ลูกบอลยังคาบเกี่ยวจุดเป้าหมาย",
               0.0, "ทฤษฎีเชิงเส้นให้ 0 (state feedback + ตัว trim integrator)"),
        Target("overshoot_pct", "ไม่พุ่งเลยเป้าหมายมากเกินไป", "overshoot", 20.0, "%",
               f"ทฤษฎีที่ ζ = {cl.zeta:.2f} ให้ {cl.overshoot_pct:.0f}% เผื่อไว้ถึง 20% สำหรับ dead zone และ delay",
               cl.overshoot_pct, f"ทฤษฎี {cl.overshoot_pct:.0f}%"),
        Target("track_mm", "ติดตามเส้นทางที่กำหนด", "mean tracking error", 0.5 * null_mm, "mm",
               f"= ครึ่งหนึ่งของความคลาดของตัวควบคุมที่ปล่อยบอลนิ่งกลางแผ่น ({null_mm:.0f} mm = รัศมีเส้นทาง) "
               f"ต้องดีกว่าการไม่ทำอะไรอย่างน้อยครึ่งหนึ่งจึงนับว่าตามเส้นทางได้จริง "
               f"ทฤษฎีเชิงเส้นให้ {path_mean_mm:.2f} mm",
               path_mean_mm, f"ทฤษฎีเชิงเส้น {path_mean_mm:.2f} mm; ตัวควบคุมที่ไม่ทำอะไร {null_mm:.0f} mm"),
        Target("success", "ความน่าเชื่อถือของระบบ", "success rate", 80.0, "%",
               "= 4 ใน 5 รอบ รายงานพร้อมจำนวนรอบที่ทดสอบเพื่อให้เห็นขนาดกลุ่มตัวอย่าง"),
    ]


def summary_lines(k_scale: float = 0.076, kv_scale: float = 2.8,
                  path_k_scale: float = 0.15, path_kv_scale: float = 1.8,
                  amplitude_m: float = 0.030, period_s: float = 20.0) -> list[str]:
    """Human-readable dump of every derived quantity (English; used by the CLI and the tests)."""
    g, derivation = plant_gain()
    lines = [f"plant: {derivation}",
             f"  measured: x {measured('plant_gain_x'):g} ({100 * measured('plant_gain_x') / g:.0f}% of theory), "
             f"y {measured('plant_gain_y'):g} ({100 * measured('plant_gain_y') / g:.0f}%)",
             f"  actuator lag model TA = {TA:.3f} s (params.py)"]
    for label, ks, kvs in (("balance", k_scale, kv_scale), ("path", path_k_scale, path_kv_scale)):
        cl = closed_loop(ks, kvs)
        pk, mean, ratio = tracking_error(ks, kvs, amplitude_m, period_s)
        pm, wc = phase_margin(ks, kvs)
        lines += [f"{label} gains k-scale {ks:g} kv-scale {kvs:g}: k1 {cl.k1:.3f} rad/m, k2 {cl.k2:.3f} rad/(m/s)",
                  f"  ideal: omega_n {cl.omega_n:.2f} rad/s, zeta {cl.zeta:.2f}, overshoot {cl.overshoot_pct:.0f}%, "
                  f"t_settle(2%) {cl.settle_2pct_s:.2f} s",
                  f"  ideal tracking of {amplitude_m * 1000:g} mm / {period_s:g} s: peak {pk:.2f} mm, "
                  f"mean {mean:.2f} mm ({100 * ratio:.1f}% of amplitude)",
                  f"  with the ESTIMATED delay {measured('loop_delay'):g} s: crossover {wc:.2f} rad/s, phase margin {pm:.0f} deg",
                  f"  (falsified hypothesis theta_dead/k1: {dead_zone_floor_mm(ks):.1f} mm floor, predicted ball "
                  f"amplitude {path_amplitude_prediction_mm(ks, amplitude_m):.1f} mm -- measured did not follow)"]
    mm_px, noise, note = sensing_floor_mm()
    lines += [f"sensing: {note}",
              f"path demand: a {amplitude_m * 1000:g} mm / {period_s:g} s lap needs only "
              f"{reference_tilt_demand_deg(amplitude_m, period_s):.3f} deg of tilt, versus a dead zone of "
              f"{measured('dead_zone_effective'):g} deg -> the ball cannot glide, it must stick and slip"]
    return lines


if __name__ == "__main__":
    for line in summary_lines():
        print(line)
    print("\ntargets:")
    for t in targets():
        limit = "" if t.key == "plant_gain" else f"{t.limit:g} {t.unit}"
        print(f"  {t.metric:<22} {limit:>10}   {t.source_th}")
