"""Path-following analysis that does not depend on timing: cross-track error and the transient/steady split.

Time-synchronous error (ball position minus the reference point at the same instant) is large when the
ball runs on the line but lags behind the reference. Cross-track error is the shortest distance from the
ball to the reference curve, so a pure lag costs nothing. Both are reported; the report says which is which.

    transient : hold phase (path_phase 1) and amplitude ramp (2) - the ball converges towards the centre
    steady    : full path (path_phase 3) - the ball is expected to be on the curve
"""

from __future__ import annotations

import numpy as np

import trajectory as TJ
from ch4_figures import HOLD_S, PERIOD_S, RAMP_S, load_run


def reference_polyline(shape: str) -> np.ndarray:
    """Closed reference curve (N+1, 2) in metres, full amplitude."""
    ref = TJ.PathReference(shape, PERIOD_S, HOLD_S, RAMP_S)
    pts = ref._curve * ref.size_m
    return np.vstack([pts, pts[:1]])


def cross_track_m(points: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """Distance (m) from each of the (M, 2) points to the closed polyline (K+1, 2)."""
    a, b = curve[:-1], curve[1:]
    ab = b - a
    denom = np.maximum(np.sum(ab * ab, axis=1), 1e-12)
    out = np.empty(len(points))
    for start in range(0, len(points), 512):
        p = points[start:start + 512][:, None, :]
        t = np.clip(np.sum((p - a) * ab, axis=2) / denom, 0.0, 1.0)
        proj = a + t[..., None] * ab
        out[start:start + 512] = np.min(np.linalg.norm(p - proj, axis=2), axis=1)
    return out


def steady_frames(stamp: str) -> dict[str, np.ndarray]:
    """Valid frames of the full-path phase, with time measured from the start of that phase."""
    d = load_run(stamp)
    sel = d["valid"] & (d["phase"] == 3)
    t = d["t_s"][sel] - d["t0"] - (HOLD_S + RAMP_S)
    return {"t": t, "x": d["x_m"][sel], "y": d["y_m"][sel], "xr": d["x_ref_m"][sel], "yr": d["y_ref_m"][sel],
            "lost": d["lost_neutralized"][sel] == 1}


def run_summary(stamp: str, shape: str) -> dict:
    f = steady_frames(stamp)
    curve = reference_polyline(shape)
    pts = np.stack([f["x"], f["y"]], axis=1)
    ct = cross_track_m(pts, curve) * 1000.0
    sync = np.hypot(f["x"] - f["xr"], f["y"] - f["yr"]) * 1000.0
    return {"cross_mm": float(ct.mean()), "cross_rms_mm": float(np.sqrt(np.mean(ct ** 2))),
            "sync_mm": float(sync.mean()), "frames": int(len(ct))}


def polar_band(stamps: list[str], shape: str, bins: int = 36):
    """Ball radius as a function of polar angle: per-run mean in each angle bin, then mean and SD over runs."""
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    per_run = []
    for s in stamps:
        f = steady_frames(s)
        ang = np.arctan2(f["y"], f["x"])
        rad = np.hypot(f["x"], f["y"]) * 1000.0
        idx = np.clip(np.digitize(ang, edges) - 1, 0, bins - 1)
        row = np.full(bins, np.nan)
        for b in range(bins):
            if np.sum(idx == b) >= 5:
                row[b] = rad[idx == b].mean()
        per_run.append(row)
    arr = np.array(per_run)
    centres = np.degrees((edges[:-1] + edges[1:]) / 2)
    return centres, np.nanmean(arr, axis=0), np.nanstd(arr, axis=0, ddof=1), np.sum(~np.isnan(arr), axis=0)


def reference_radius_mm(shape: str, angles_deg: np.ndarray) -> np.ndarray:
    curve = reference_polyline(shape)[:-1] * 1000.0
    ang = np.degrees(np.arctan2(curve[:, 1], curve[:, 0]))
    rad = np.hypot(curve[:, 0], curve[:, 1])
    order = np.argsort(ang)
    return np.interp(angles_deg, ang[order], rad[order], period=360.0)


def transient_curves(stamps: list[str], secs: float = 11.0, dt: float = 0.1):
    """Distance from the centre against time for the hold + ramp phases (the ball converging), all runs on a common grid."""
    grid = np.arange(0.0, secs, dt)
    rows = []
    for s in stamps:
        d = load_run(s)
        v = d["valid"] & (d["phase"] <= 2)
        t = d["t_s"][v] - d["t0"]
        if len(t) < 20:
            continue
        r = np.hypot(d["x_m"][v] - d["x_ref_m"][v], d["y_m"][v] - d["y_ref_m"][v]) * 1000.0
        rows.append(np.interp(grid, t, r, left=np.nan, right=np.nan))
    arr = np.array(rows)
    return grid, np.nanmean(arr, axis=0), np.nanstd(arr, axis=0, ddof=1), arr
