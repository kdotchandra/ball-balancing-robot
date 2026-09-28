"""Stability analysis of the balance loop, from the model the LQR gain was designed on.

The design model of ONE axis (x or y are identical) has three states: the ball's position error e,
its velocity error de, and the plate tilt theta_a (a first-order lag of the commanded tilt):

    e'       = de
    de'      = g_eff * theta_a           g_eff = KR*G = 5.886 m/s^2/rad  (rolling ball, params.py)
    theta_a' = (u - theta_a) / TA        TA = 0.181 s                    (params.py)
    u        = -K [e, de, theta_a]       K = K_AXIS * [k, k*kv, k]       (state feedback, LQR)

Everything below is exact linear analysis of that model: poles of the closed loop, the
Routh-Hurwitz condition, gain / phase margin with a pure delay added on the command, the root locus
against the gain, and the time response. Nothing here needs the real plate.

The one number the model does not contain is the extra pure delay between "the ball moved" and "the
plate reacted" (camera latency + image processing + servo command period). It is the main unknown
of the stability picture, so it is always a PARAMETER here (DEFAULT_DELAY_S is a stated estimate,
DELAY_RANGE_S the range it is expected to lie in), never presented as a measured constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from params import K_AXIS, KR, G, TA

G_EFF = KR * G
DEFAULT_DELAY_S = 0.13            # estimate: about 3 camera frames + servo command period
DELAY_RANGE_S = (0.05, 0.25)      # range it is expected to lie in


@dataclass
class Model:
    """One axis of the design model. With k_i > 0 an integral state of the position error is added.

    States: [e, de, theta_a] (three), or [xi, e, de, theta_a] (four) when k_i > 0, with xi' = e and
    u = -(k_i*xi + k1*e + k2*de + k3*theta_a). The integral is the LQI-style term used while a path is
    being followed (main.py, --path-ki); the other three gains are the unchanged balance gains.
    """
    k_scale: float
    kv_scale: float
    k_i: float = 0.0

    @property
    def n(self) -> int:
        return 4 if self.k_i > 0.0 else 3

    @property
    def e_index(self) -> int:
        """Position of the position-error state in the state vector."""
        return self.n - 3

    @property
    def K(self) -> np.ndarray:
        """The controller gain exactly as main.py builds it: K_AXIS * [k, k*kv, k] (plus k_i in front)."""
        k3 = np.array([K_AXIS[0] * self.k_scale, K_AXIS[1] * self.k_scale * self.kv_scale,
                       K_AXIS[2] * self.k_scale])
        return np.concatenate([[self.k_i], k3]) if self.n == 4 else k3

    @property
    def A(self) -> np.ndarray:
        core = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, G_EFF], [0.0, 0.0, -1.0 / TA]])
        if self.n == 3:
            return core
        full = np.zeros((4, 4))
        full[0, 1] = 1.0                      # xi' = e
        full[1:, 1:] = core
        return full

    @property
    def B(self) -> np.ndarray:
        b = np.array([0.0, 0.0, 1.0 / TA])
        return b if self.n == 3 else np.concatenate([[0.0], b])

    @property
    def A_closed(self) -> np.ndarray:
        return self.A - np.outer(self.B, self.K)


# ------------------------------------------------------------------ poles and Routh-Hurwitz
def poles(model: Model) -> np.ndarray:
    return np.linalg.eigvals(model.A_closed)


def characteristic_polynomial(model: Model) -> np.ndarray:
    """[1, a2, a1, a0] of  s^3 + a2 s^2 + a1 s + a0 = det(sI - (A - B K))."""
    return np.real(np.poly(model.A_closed))


def routh_first_column(coeffs: np.ndarray) -> list[float]:
    """First column of the Routh array of a polynomial (any degree); no sign change = stable."""
    coeffs = [float(c) for c in coeffs]
    rows = [coeffs[0::2], coeffs[1::2]]
    width = len(rows[0])
    rows[1] += [0.0] * (width - len(rows[1]))
    for _ in range(len(coeffs) - 2):
        a, b = rows[-2], rows[-1]
        pivot = b[0] if abs(b[0]) > 1e-12 else 1e-12
        new = [(pivot * a[i + 1] - a[0] * b[i + 1]) / pivot for i in range(width - 1)] + [0.0]
        rows.append(new)
    return [r[0] for r in rows]


def routh_table(model: Model) -> dict:
    """Routh array of the third-order characteristic polynomial and the verdict.

    For s^3 + a2 s^2 + a1 s + a0 the array is
        s^3 | 1                  a1
        s^2 | a2                 a0
        s^1 | (a2*a1 - a0)/a2    0
        s^0 | a0
    and the system is stable exactly when the first column has no sign change,
    i.e. a2 > 0, a0 > 0 and a2*a1 > a0.
    """
    _, a2, a1, a0 = characteristic_polynomial(model)
    b1 = (a2 * a1 - a0) / a2
    column = [1.0, a2, b1, a0]
    return {"a2": a2, "a1": a1, "a0": a0, "b1": b1, "first_column": column,
            "stable": all(c > 0 for c in column), "condition": a2 * a1 - a0}


# ------------------------------------------------------------------ frequency response and margins
def loop_response(model: Model, w: float | np.ndarray, delay_s: float = 0.0) -> np.ndarray:
    """L(jw) = K (jwI - A)^-1 B e^{-jw*delay}: the loop broken at the plate command."""
    w_arr = np.atleast_1d(np.asarray(w, dtype=float))
    out = np.empty(len(w_arr), dtype=complex)
    for i, wi in enumerate(w_arr):
        s = 1j * wi
        out[i] = model.K @ np.linalg.solve(s * np.eye(model.n) - model.A, model.B) * np.exp(-s * delay_s)
    return out if np.ndim(w) else out[0]


def margins(model: Model, delay_s: float = 0.0) -> dict:
    """Phase margin (deg), gain margin (dB), and the frequencies where they are measured.

    `stable` is decided by the true closed-loop poles of the delayed loop (a Pade approximation of
    the delay), not just by the sign of the margins, because a plot of margins alone can mislead
    when there are several crossovers.
    """
    ws = np.logspace(-2.0, 1.5, 7000)
    ls = loop_response(model, ws, delay_s)
    mag = np.abs(ls)
    phase = np.degrees(np.unwrap(np.angle(ls)))
    cross = np.where(np.diff(np.sign(mag - 1.0)))[0]
    pm = wc = None
    if len(cross):
        i = cross[-1]                                   # the highest crossover is the one that limits stability
        # principal-value phase margin: the plain unwrapped phase is off by 360 deg when an integrator is present
        pm, wc = float(((180.0 + np.degrees(np.angle(ls[i])) + 180.0) % 360.0) - 180.0), float(ws[i])
    pcross = np.where(np.diff(np.sign(phase + 180.0)))[0] if model.n == 3 else []
    gm = wpc = None
    if len(pcross):
        i = pcross[0]
        gm, wpc = float(-20.0 * math.log10(mag[i])), float(ws[i])
    return {"pm_deg": pm, "wc": wc, "gm_db": gm, "w_pc": wpc, "stable": is_stable(model, delay_s)}


def critical_delay(model: Model, upper_s: float = 1.5) -> float:
    """Smallest pure delay that makes the closed loop unstable (bisection on the pole locations)."""
    lo, hi = 0.0, upper_s
    if not is_stable(model, 0.0):
        return 0.0
    if is_stable(model, hi):
        return float("inf")
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if is_stable(model, mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------ root locus (with a Pade delay)
def _pade(delay_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Second-order Pade approximation of e^{-s*delay}: (num, den) as polynomials in s."""
    if delay_s <= 0:
        return np.array([1.0]), np.array([1.0])
    t = delay_s
    return np.array([t * t / 12.0, -t / 2.0, 1.0]), np.array([t * t / 12.0, t / 2.0, 1.0])


def closed_loop_poles_with_delay(model: Model, delay_s: float = 0.0) -> np.ndarray:
    """Poles of the loop with the delay replaced by its Pade approximation.

    With N(s) = det(sI-A+BK) - det(sI-A) = K adj(sI-A) B  and  D(s) = det(sI-A), the loop is
    L = N/D * P_num/P_den, and the closed loop satisfies  D*P_den + N*P_num = 0.
    """
    den = np.poly(model.A)
    num = np.poly(model.A_closed) - den
    p_num, p_den = _pade(delay_s)
    poly = np.polyadd(np.polymul(den, p_den), np.polymul(num, p_num))
    return np.roots(poly)


def is_stable(model: Model, delay_s: float = 0.0) -> bool:
    return bool(np.all(np.real(closed_loop_poles_with_delay(model, delay_s)) < 0.0))


def root_locus(kv_scale: float, k_scales: np.ndarray, delay_s: float = 0.0) -> np.ndarray:
    """Closed-loop poles for each gain scale, the velocity gain following at fixed ratio kv_scale."""
    return np.array([closed_loop_poles_with_delay(Model(float(k), kv_scale), delay_s) for k in k_scales])


def critical_gain_scale(kv_scale: float, delay_s: float, lo: float = 0.005, hi: float = 2.0) -> float:
    """Gain scale at which the loop turns unstable (first crossing of the imaginary axis)."""
    if not is_stable(Model(lo, kv_scale), delay_s):
        return 0.0
    if is_stable(Model(hi, kv_scale), delay_s):
        return float("inf")
    for _ in range(50):
        mid = math.sqrt(lo * hi)
        if is_stable(Model(mid, kv_scale), delay_s):
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


# ------------------------------------------------------------------ time response
def step_response(model: Model, e0_m: float, secs: float = 12.0, dt: float = 0.005,
                  delay_s: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Error e(t) after releasing the ball e0 metres from the target, at rest, plate level.

    Simulated with a delay buffer on the command (exact for any delay, no Pade needed).
    """
    steps = int(secs / dt)
    lag = int(round(delay_s / dt))
    x = np.zeros(model.n)
    x[model.e_index] = e0_m
    history = [0.0] * max(lag, 0)
    T = np.arange(steps) * dt
    E = np.empty(steps)
    K, A, B = model.K, model.A, model.B
    for i in range(steps):
        E[i] = x[model.e_index]
        u = float(-K @ x)
        history.append(u)
        u_applied = history.pop(0) if lag else u
        x = x + (A @ x + B * u_applied) * dt
    return T, E


def step_metrics(model: Model, e0_m: float = 0.06, delay_s: float = 0.0, zone_m: float = 0.02) -> dict:
    """Overshoot (% of the start distance) and 2%-settling time of the response above."""
    T, E = step_response(model, e0_m, secs=30.0, delay_s=delay_s)
    overshoot = max(0.0, -float(E.min())) / e0_m * 100.0
    outside = np.where(np.abs(E) > 0.02 * e0_m)[0]
    settle = float(T[outside[-1]]) if len(outside) else 0.0
    return {"overshoot_pct": overshoot, "settle_2pct_s": settle, "steady_state_m": float(abs(E[-1]))}


# ------------------------------------------------------------------ following a moving reference
def tracking_error_amplitude(model: Model, amplitude_m: float, period_s: float, delay_s: float = 0.0) -> float:
    """Peak position error (m) following r = A sin(wt), from the frequency response e/r''.

    The controller uses the velocity ERROR (ball velocity minus reference velocity), so the error
    states obey  x' = A x + B u(t-d) + [0, -r'', 0], and |E| = |G_ed(jw)| * A w^2 with
    G_ed = [1 0 0] (jwI - A + B K e^{-jwd})^-1 [0 -1 0]^T.
    """
    w = 2.0 * math.pi / period_s
    s = 1j * w
    M = s * np.eye(model.n) - model.A + np.outer(model.B, model.K) * np.exp(-s * delay_s)
    disturbance = np.zeros(model.n)
    disturbance[model.e_index + 1] = -1.0          # the reference acceleration enters the velocity-error equation
    gain = abs(np.linalg.solve(M, disturbance)[model.e_index])
    return float(gain * amplitude_m * w * w)


def simulate_reference(model: Model, reference, secs: float, dt: float = 0.01, delay_s: float = 0.0,
                       integral_from_s: float = 0.0):
    """Both axes of the linear model following a reference function t -> (position, velocity) in metres.

    Returns time, ball position (N,2) and reference position (N,2). Axes are independent, so this is
    just the single-axis model run twice. When the model has an integral gain (k_i > 0) the integral of
    the position error runs from `integral_from_s` on (in the real controller: from the start of the
    ramp), the same way main.py does it.
    """
    lag = int(round(delay_s / dt))
    k3 = model.K[-3:]
    k_i = model.k_i
    n = int(secs / dt)
    pos = np.zeros((n, 2))
    ref_out = np.zeros((n, 2))
    T = np.arange(n) * dt
    theta = [0.0, 0.0]
    xi = [0.0, 0.0]
    hist = [[0.0] * max(lag, 0), [0.0] * max(lag, 0)]
    velocity = [0.0, 0.0]
    position = [0.0, 0.0]
    for i in range(n):
        r, rd = reference(T[i])
        for j in range(2):
            err = position[j] - r[j]
            derr = velocity[j] - rd[j]
            u = float(-(k3 @ np.array([err, derr, theta[j]]) + (k_i * xi[j] if T[i] >= integral_from_s else 0.0)))
            hist[j].append(u)
            u_applied = hist[j].pop(0) if lag else u
            theta_prev = theta[j]
            theta[j] += (u_applied - theta[j]) / TA * dt
            velocity[j] += G_EFF * theta_prev * dt
            position[j] += velocity[j] * dt
            if k_i > 0.0 and T[i] >= integral_from_s:
                xi[j] += err * dt
            pos[i, j] = position[j]
            ref_out[i, j] = r[j]
    return T, pos, ref_out


if __name__ == "__main__":
    for name, k, kv in (("balance (k 0.076, kv 2.8)", 0.076, 2.8), ("path (k 0.15, kv 1.8)", 0.15, 1.8)):
        m = Model(k, kv)
        print(f"\n{name}: K = {np.round(m.K, 3)}")
        print("  poles:", np.round(poles(m), 3))
        r = routh_table(m)
        print(f"  Routh: a2={r['a2']:.3f} a1={r['a1']:.3f} a0={r['a0']:.3f}  a2*a1-a0={r['condition']:.3f} -> {'stable' if r['stable'] else 'UNSTABLE'}")
        for d in (0.0, DEFAULT_DELAY_S, 0.25):
            mg = margins(m, d)
            print(f"  delay {d:.2f} s: PM {mg['pm_deg']:.1f} deg at {mg['wc']:.2f} rad/s, GM {mg['gm_db'] if mg['gm_db'] is None else round(mg['gm_db'], 1)} dB, stable={mg['stable']}")
        print(f"  critical delay {critical_delay(m):.3f} s;  critical gain scale at 0.13 s: {critical_gain_scale(kv, 0.13):.3f}")
        sm = step_metrics(m)
        print(f"  step from 6 cm: overshoot {sm['overshoot_pct']:.1f}%, settle {sm['settle_2pct_s']:.2f} s")
