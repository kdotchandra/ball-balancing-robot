"""Figures for chapter 4 of the project report (Thai labels, one message per figure).

Every builder returns a `Figure`: the PNG path, its number and Thai title, a one-line Thai caption
saying what is plotted, and a Thai "how to read it" line saying what the reader should take away.
ch4_docx.py lays them out in that order, so the text never drifts from the picture.

Colours are fixed across the whole report:
    blue   = simulation (the design model, computed with stability.py)
    red    = measured on the real platform
    orange = the reference / target the ball is asked to follow

Which runs feed the figures: the counted trials of the tags balance / circle / hexagon, found through
the sidecar files (logs/run_<stamp>.json, written by main.py). Before those trials exist the figures
fall back to the hand-listed logs in report_runs.json, so the report can always be built.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

import experiment_profile as PROF
import run_metrics as RM
import stability as ST
import theory_limits as TH
import trajectory as TJ
from params import TA
from test_tilt_direction import _lagged_double_integral

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
FIG_DIR = LOGS / "figures"
MANIFEST = ROOT / "report_runs.json"
FONT_FILE = ROOT / "assets" / "fonts" / "NotoSansThai.ttf"

C_SIM = "#1f77b4"
C_REAL = "#d62728"
C_REF = "#ff7f0e"
C_GREY = "#7f7f7f"
C_OK = "#2ca02c"

PERIOD_S = PROF.PATH_PERIOD_S
HOLD_S = PROF.PATH_HOLD_S
RAMP_S = PROF.PATH_RAMP_S
CAMERA_LIMIT_CM = 9.0

LABELS = {"balance": "ทรงตัวที่จุดศูนย์กลาง", "circle": "วงกลม รัศมี 3 cm",
          "hexagon": "หกเหลี่ยมที่ครอบวงกลมรัศมี 3 cm"}

font_manager.fontManager.addfont(str(FONT_FILE))
plt.rcParams["font.family"] = ["Noto Sans Thai", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11


@dataclass
class Figure:
    key: str
    number: str
    path: Path
    title: str
    caption: str
    reading: str
    notes: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
def manifest() -> dict:
    """The run groups. Tagged trials (sidecars) win; the hand-listed logs are the fallback."""
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for tag in ("balance", "circle", "hexagon"):
        paths = RM.collect_runs(tag)
        if paths:
            meta = RM.read_meta(paths[0]) or {}
            m[tag] = {"label": LABELS[tag], "gains": meta.get("gains", [PROF.K_SCALE, PROF.KV_SCALE]),
                      "k_i": float(meta.get("path_ki", PROF.PATH_KI)) if tag != "balance" else 0.0,
                      "runs": [p.stem[len("run_"):] for p in paths], "from_tags": True}
        else:
            m.setdefault(tag, {"label": LABELS[tag], "gains": [PROF.K_SCALE, PROF.KV_SCALE], "runs": []})
            m[tag]["from_tags"] = False
            m[tag].setdefault("k_i", 0.0)
    m["ladder_tags"] = sorted({(RM.read_meta(p.with_suffix(".csv")) or {}).get("tag", "")
                               for p in LOGS.glob("run_*.json")
                               if str((RM.read_meta(p.with_suffix(".csv")) or {}).get("tag", "")).endswith(("ladder_0.05", "ladder_0.10", "ladder_0.20"))})
    return m


def balance_gains(m: dict) -> tuple[float, float]:
    return tuple(m["balance"]["gains"])


def chosen_k_i(m: dict) -> float:
    """The integral gain the path trials ran with (from their sidecars), else the profile value."""
    return float(m["circle"].get("k_i", PROF.PATH_KI))


def load_run(stamp: str) -> dict[str, np.ndarray]:
    with open(LOGS / f"run_{stamp}.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    def col(name: str, default: float = float("nan")) -> np.ndarray:
        return np.array([float(r[name]) if r.get(name, "") not in ("", None) else default for r in rows])

    data = {k: col(k) for k in ("t_s", "control_valid", "x_m", "y_m", "x_ref_m", "y_ref_m", "lost_neutralized")}
    data["phase"] = col("path_phase", 0.0) if "path_phase" in rows[0] else np.zeros(len(rows))
    valid = data["control_valid"] == 1
    data["valid"] = valid
    data["t0"] = float(data["t_s"][valid][0]) if valid.any() else 0.0
    return data


def balance_curve(stamp: str):
    """(time since first detection [s], distance from the target [mm]) of one balance run."""
    d = load_run(stamp)
    v = d["valid"]
    t = d["t_s"][v] - d["t0"]
    r = np.hypot(d["x_m"][v] - d["x_ref_m"][v], d["y_m"][v] - d["y_ref_m"][v]) * 1000.0
    return t, r


def start_info(stamp: str) -> tuple[float, float]:
    """(distance in cm, angle in degrees) where the ball was first tracked, relative to the target."""
    d = load_run(stamp)
    v = d["valid"]
    x = (d["x_m"][v] - d["x_ref_m"][v])[:5].mean() * 100.0
    y = (d["y_m"][v] - d["y_ref_m"][v])[:5].mean() * 100.0
    return float(math.hypot(x, y)), float(math.degrees(math.atan2(y, x)))


def path_reference(shape: str):
    ref = TJ.PathReference(shape, PERIOD_S, HOLD_S, RAMP_S)

    def fn(t: float):
        x, y, vx, vy = ref.at(t)
        return np.array([x, y]), np.array([vx, vy])

    return fn


def sim_path(shape: str, gains, k_i: float = 0.0, secs: float = 60.0, delay_s: float | None = None):
    delay_s = ST.DEFAULT_DELAY_S if delay_s is None else delay_s
    model = ST.Model(gains[0], gains[1], k_i)
    return ST.simulate_reference(model, path_reference(shape), secs=secs, dt=0.01, delay_s=delay_s,
                                 integral_from_s=HOLD_S)


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------------------------
# 4.1 stability
# ----------------------------------------------------------------------------------------------
def fig_poles(m: dict) -> Figure:
    g = balance_gains(m)
    k_i = chosen_k_i(m) if chosen_k_i(m) > 0 else 0.10
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.axvspan(0, 1, color="#f4cccc", alpha=0.5)
    ax.axvspan(-8, 0, color="#d9ead3", alpha=0.45)
    ax.text(-7.6, 3.2, "ครึ่งซ้าย: เสถียร", color="#38761d", fontsize=10)
    ax.text(0.05, 3.2, "ครึ่งขวา:\nไม่เสถียร", color="#990000", fontsize=10)
    p3 = ST.poles(ST.Model(*g))
    ax.plot(p3.real, p3.imag, "o", color=C_SIM, ms=11, label="เกนชุดทรงตัว (ไม่มีตัวอินทิเกรต)")
    p4 = ST.poles(ST.Model(g[0], g[1], k_i))
    ax.plot(p4.real, p4.imag, "s", mfc="none", mec=C_REF, ms=11, mew=2,
            label=f"เพิ่มตัวอินทิเกรต k_i = {k_i:g}")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=1.5)
    ax.set_xlim(-8, 1)
    ax.set_ylim(-3.6, 3.8)
    ax.set_xlabel("ส่วนจริงของโพล (1/วินาที)  ยิ่งซ้ายยิ่งตอบสนองเร็วและนิ่ง")
    ax.set_ylabel("ส่วนจินตภาพ (rad/วินาที)")
    ax.set_title("ตำแหน่งโพลของระบบวงปิด")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    return Figure("poles", "4.1", _save(fig, "fig4_01_poles.png"),
                  "ตำแหน่งโพลของระบบวงปิด (ยังไม่รวมเวลาหน่วง)",
                  "จุดทึบคือระบบเดิม จุดกลวงคือระบบที่เพิ่มตัวอินทิเกรตช่วงเดินตามเส้นทาง",
                  "โพลทุกตัวอยู่ครึ่งซ้ายของระนาบ แปลว่าระบบเสถียร การเพิ่มตัวอินทิเกรตเพิ่มโพลอีกหนึ่งตัวใกล้แกนตั้ง "
                  "(ตอบสนองช้า) แต่ไม่ได้ผลักโพลเดิมออกจากครึ่งซ้าย")


def fig_margin_vs_delay(m: dict) -> Figure:
    g = balance_gains(m)
    k_sel = chosen_k_i(m) if chosen_k_i(m) > 0 else 0.10
    delays = np.linspace(0.0, 0.40, 81)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    lo, hi = ST.DELAY_RANGE_S
    ax.axvspan(lo, hi, color="#fff2cc", alpha=0.9, label=f"ช่วงเวลาหน่วงที่คาดว่าเป็น ({lo:.2f}–{hi:.2f} s)")
    for k_i, colour, ls, label in ((0.0, C_SIM, "-", "ไม่มีตัวอินทิเกรต"),
                                   (k_sel, C_REF, "--", f"k_i = {k_sel:g}"),
                                   (0.5, C_GREY, ":", "k_i = 0.5 (สูงเกิน)")):
        model = ST.Model(g[0], g[1], k_i)
        pm = [ST.margins(model, d)["pm_deg"] for d in delays]
        ax.plot(delays, pm, color=colour, ls=ls, lw=2.2, label=label)
    ax.axhline(0, color="black", lw=1.2)
    ax.axhline(30, color=C_GREY, ls=":", lw=1.0)
    ax.text(0.395, 31.5, "เกณฑ์ทั่วไป 30°", color=C_GREY, fontsize=8, ha="right")
    ax.text(0.395, -7, "ต่ำกว่าเส้นนี้ = ไม่เสถียร", fontsize=8, ha="right", color="#990000")
    ax.set_xlabel("เวลาหน่วงเพิ่มในลูป (วินาที)")
    ax.set_ylabel("Phase margin (องศา)")
    ax.set_title("ขอบเขตเสถียรภาพเทียบกับเวลาหน่วง")
    ax.set_xlim(0, 0.4)
    ax.set_ylim(-10, 55)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    return Figure("margin_delay", "4.2", _save(fig, "fig4_02_margin_delay.png"),
                  "Phase margin ที่ลดลงเมื่อเวลาหน่วงในลูปเพิ่มขึ้น และเมื่อเพิ่มตัวอินทิเกรต",
                  "แถบสีเหลืองคือช่วงที่คาดว่าเวลาหน่วงจริงอยู่ เส้นต่างแบบคือค่า k_i ต่างกัน",
                  "ยิ่งเวลาหน่วงมาก ระบบยิ่งมีที่เผื่อน้อยลงและใกล้ไม่เสถียร ตัวอินทิเกรตขนาดเล็กแทบไม่ทำให้ที่เผื่อลดลง "
                  "แต่ค่าสูงเกินไป (k_i = 0.5) ทำให้ที่เผื่อหายไปมาก จึงจำกัด k_i ไม่เกิน 0.3")


def fig_root_locus(m: dict) -> Figure:
    g = balance_gains(m)
    ks = np.concatenate([np.linspace(0.005, 0.12, 60), np.linspace(0.12, 0.6, 120)])
    delay = ST.DEFAULT_DELAY_S
    locus = ST.root_locus(g[1], ks, delay)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.axvspan(0, 4, color="#f4cccc", alpha=0.5)
    ax.axvspan(-8, 0, color="#d9ead3", alpha=0.35)
    for j in range(locus.shape[1]):
        ax.plot(locus[:, j].real, locus[:, j].imag, ".", ms=2.5, color=C_SIM, alpha=0.8)
    used = g[0]
    p_used = ST.closed_loop_poles_with_delay(ST.Model(used, g[1]), delay)
    ax.plot(p_used.real, p_used.imag, "o", ms=9, color=C_REAL, label=f"เกนที่ใช้ (k={used:g})")
    kc = ST.critical_gain_scale(g[1], delay)
    p_crit = ST.closed_loop_poles_with_delay(ST.Model(kc, g[1]), delay)
    p_crit = p_crit[np.abs(p_crit.real) < 0.05]
    ax.plot(p_crit.real, p_crit.imag, "X", ms=11, color="black", label=f"เริ่มไม่เสถียรที่ k = {kc:.2f}")
    ax.axvline(0, color="black", lw=1.5)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlim(-8, 2.5)
    ax.set_ylim(-9, 9)
    ax.set_xlabel("ส่วนจริง (1/วินาที)")
    ax.set_ylabel("ส่วนจินตภาพ (rad/วินาที)")
    ax.set_title(f"Root locus เมื่อเพิ่มเกน (เวลาหน่วง {delay:g} s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    return Figure("root_locus", "4.3", _save(fig, "fig4_03_root_locus.png"),
                  "เส้นทางการเคลื่อนที่ของโพลเมื่อเพิ่มเกนของตัวควบคุมจาก 0 ขึ้นไป",
                  "จุดสีน้ำเงินคือโพลที่เกนค่าต่างๆ จุดแดงคือเกนที่ใช้จริง กากบาทดำคือจุดที่โพลข้ามเข้าครึ่งขวา (เริ่มไม่เสถียร)",
                  f"เกนที่ใช้จริงอยู่ห่างจากจุดไม่เสถียรพอสมควร (เกนที่เริ่มไม่เสถียรมากกว่าที่ใช้ประมาณ {kc / used:.1f} เท่า) "
                  "แต่เมื่อเพิ่มเกนต่อไปโพลคู่หนึ่งจะเคลื่อนไปทางขวาและระบบเริ่มแกว่งจนไม่เสถียร")


# ----------------------------------------------------------------------------------------------
# 4.2 simulation
# ----------------------------------------------------------------------------------------------
def fig_sim_response(m: dict, start_mm: float = 60.0) -> Figure:
    model = ST.Model(*balance_gains(m))
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    T, E = ST.step_response(model, start_mm / 1000.0, secs=10.0, delay_s=0.0)
    ax.plot(T, np.abs(E) * 1000.0, color=C_SIM, lw=2.4, label="ไม่มีเวลาหน่วง")
    T, E2 = ST.step_response(model, start_mm / 1000.0, secs=10.0, delay_s=ST.DEFAULT_DELAY_S)
    ax.plot(T, np.abs(E2) * 1000.0, color=C_SIM, lw=2.0, ls="--", label=f"มีเวลาหน่วง {ST.DEFAULT_DELAY_S:g} s")
    ax.axhline(40.0, color=C_GREY, ls=":", lw=1.0)
    ax.text(9.9, 42, "โซนศูนย์กลาง 4 cm", color=C_GREY, fontsize=8, ha="right")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("เวลา (วินาที)")
    ax.set_ylabel("ระยะห่างจากจุดเป้าหมาย (mm)")
    ax.set_title(f"ผลการจำลอง: ปล่อยลูกบอลห่างจากศูนย์กลาง {start_mm:g} mm")
    ax.set_xlim(0, 10)
    ax.set_ylim(-2, start_mm * 1.08)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    return Figure("sim_response", "4.4", _save(fig, "fig4_04_sim_response.png"),
                  "การจำลองการเข้าสู่ศูนย์กลางด้วยแบบจำลองเชิงเส้น (Python)",
                  "ระยะห่างจากศูนย์กลางเทียบกับเวลา เส้นทึบคือกรณีอุดมคติ เส้นประคือกรณีเพิ่มเวลาหน่วงในลูป",
                  "ลูกบอลเข้าสู่ศูนย์กลางแล้วนิ่ง (ความคลาดขณะนิ่งเป็นศูนย์) การเพิ่มเวลาหน่วงทำให้เลยเป้าเล็กน้อยแต่ยังเสถียร")


def fig_sim_path(m: dict) -> Figure:
    g, k_i = balance_gains(m), chosen_k_i(m)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8))
    lines = []
    for ax, shape in zip(axes, ("circle", "hexagon")):
        T, P, R = sim_path(shape, g, k_i)
        sel = T > HOLD_S + RAMP_S
        ref = TJ.PathReference(shape, PERIOD_S, HOLD_S, RAMP_S)
        ts = np.linspace(HOLD_S + RAMP_S, HOLD_S + RAMP_S + PERIOD_S, 400)
        pts = np.array([ref.at(t)[:2] for t in ts]) * 1000
        ax.plot(pts[:, 0], pts[:, 1], "--", color=C_REF, lw=2.0, label="เส้นทางที่สั่ง")
        ax.plot(P[sel, 0] * 1000, P[sel, 1] * 1000, color=C_SIM, lw=1.8, label="ลูกบอล (จำลอง)")
        ax.set_title(LABELS[shape], fontsize=10)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect("equal")
        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
        lines.append(f"{LABELS[shape]} {np.hypot(*(P[sel] - R[sel]).T).mean() * 1000:.1f} mm")
    return Figure("sim_path", "4.5", _save(fig, "fig4_05_sim_path.png"),
                  "การจำลองการเดินตามเส้นทางวงกลมและหกเหลี่ยม (คาบ 20 วินาทีต่อรอบ)",
                  "เส้นประส้มคือเส้นทางที่สั่ง เส้นน้ำเงินคือเส้นทางของลูกบอลที่จำลองได้ ความคลาดเฉลี่ย: " + ", ".join(lines),
                  "ตามแบบจำลอง ลูกบอลเดินทับเส้นทางที่สั่งเกือบสนิททั้งสองรูปทรง เพราะเส้นทางเคลื่อนช้าและต้องการมุมเอียงน้อยมาก")


# ----------------------------------------------------------------------------------------------
# 4.3 real platform
# ----------------------------------------------------------------------------------------------
def fig_real_balance(m: dict) -> Figure:
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    runs = m["balance"]["runs"]
    starts = [start_info(s)[0] for s in runs] if runs else [0]
    norm = plt.Normalize(min(starts), max(max(starts), min(starts) + 1))
    cmap = plt.get_cmap("viridis_r")
    for stamp, start in zip(runs, starts):
        t, r = balance_curve(stamp)
        if len(t) < 30:
            continue
        ax.plot(t, r, lw=1.1, color=cmap(norm(start)), alpha=0.9)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("ระยะเริ่มต้นของแต่ละรอบ (cm)")
    ax.axhline(40.0, color=C_GREY, ls=":", lw=1.0)
    ax.text(29.8, 43, "โซนศูนย์กลาง 4 cm", color=C_GREY, fontsize=8, ha="right")
    ax.set_xlabel("เวลานับจากตรวจพบลูกบอลครั้งแรก (วินาที)")
    ax.set_ylabel("ระยะห่างจากจุดเป้าหมาย (mm)")
    ax.set_title(f"ผลจริง: การทรงตัวที่จุดศูนย์กลาง {len(runs)} รอบ วางลูกบอลอิสระ")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 110)
    ax.grid(alpha=0.3)
    return Figure("real_balance", "4.6", _save(fig, "fig4_06_real_balance.png"),
                  "ระยะห่างจากศูนย์กลางเทียบกับเวลา ของการทดลองจริงทุกรอบ",
                  "แต่ละเส้นคือหนึ่งรอบการทดลอง สีบอกระยะที่วางลูกบอลตอนเริ่ม (วางอิสระ ไม่ได้กำหนดจุด)",
                  "ดูว่าเส้นลดลงเข้าใกล้ศูนย์กลางหรือไม่ และเส้นสีเหลือง (เริ่มไกล) ทำได้ต่างจากเส้นสีม่วง (เริ่มใกล้) อย่างไร")


def fig_start_vs_result(m: dict) -> Figure:
    zone = RM.default_zone_cm()
    rows = []
    for stamp in m["balance"]["runs"]:
        r = RM.score_run(LOGS / f"run_{stamp}.csv", zone)
        start, _ = start_info(stamp)
        rows.append({"start": start, "lost": bool(r.get("lost")) or r.get("kind") == "short",
                     "settle": r.get("settle_s", float("nan")), "steady": r.get("steady_mm", float("nan")),
                     "success": bool(r.get("success"))})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    for ax, key, ylabel, top in ((axes[0], "settle", "เวลาเข้าโซนศูนย์กลาง 4 cm (วินาที)", 12.0),
                                 (axes[1], "steady", "ความคลาดขณะนิ่ง (mm)", 40.0)):
        ax.axvspan(CAMERA_LIMIT_CM, 12, color="#f4cccc", alpha=0.5)
        ax.text(CAMERA_LIMIT_CM + 0.1, top * 0.93, "กล้องเริ่มจับ\nไม่ค่อยได้ (>9 cm)", fontsize=8, color="#990000", va="top")
        for r in rows:
            if r["lost"]:
                ax.plot(r["start"], top * 0.97, "X", color=C_REAL, ms=11, mew=1.5)
            elif r[key] == r[key]:
                ax.plot(r["start"], min(r[key], top), "o", color=C_OK if r["success"] else "#ff7f0e", ms=9)
        ax.set_xlabel("ระยะที่วางลูกบอลตอนเริ่ม (cm)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 11.5)
        ax.set_ylim(0, top)
        ax.grid(alpha=0.3)
    axes[0].axhline(7.0, color=C_GREY, ls=":", lw=1.0)
    axes[0].text(0.15, 7.2, "เป้าหมาย 7 วินาที", fontsize=8, color=C_GREY)
    axes[1].axhline(20.0, color=C_GREY, ls=":", lw=1.0)
    axes[1].text(0.15, 20.8, "เป้าหมาย 20 mm (รัศมีลูกบอล)", fontsize=8, color=C_GREY)
    axes[0].plot([], [], "o", color=C_OK, label="สำเร็จ")
    axes[0].plot([], [], "X", color=C_REAL, label="ลูกบอลหลุด")
    axes[0].legend(fontsize=9, loc="upper left")
    n_far = sum(1 for r in rows if r["start"] > CAMERA_LIMIT_CM)
    return Figure("start_vs_result", "4.7", _save(fig, "fig4_07_start_vs_result.png"),
                  "ผลของการทรงตัวเทียบกับระยะที่วางลูกบอลตอนเริ่ม",
                  f"แต่ละจุดคือหนึ่งรอบ ตำแหน่งเริ่มวางอิสระ (มี {n_far} รอบที่เริ่มไกลกว่า 9 cm) "
                  "กากบาทแดงคือรอบที่ลูกบอลหลุด แถบสีแดงคือย่านที่กล้องเคยตรวจจับลูกบอลไม่ค่อยได้",
                  "ถ้าจุดสำเร็จอยู่ทั้งใกล้และไกล แปลว่าตัวควบคุมรับจุดเริ่มต้นไกลได้ แต่ถ้ากากบาทแดงกระจุกอยู่ในแถบสีแดง "
                  "แปลว่าที่ตกเกิดจากขีดจำกัดของกล้อง ไม่ใช่จากตัวควบคุม")


def fig_real_paths(m: dict) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.0))
    for ax, key in zip(axes, ("circle", "hexagon")):
        ref = TJ.PathReference(key, PERIOD_S, HOLD_S, RAMP_S)
        ts = np.linspace(HOLD_S + RAMP_S, HOLD_S + RAMP_S + PERIOD_S, 400)
        pts = np.array([ref.at(t)[:2] for t in ts]) * 1000
        ax.plot(pts[:, 0], pts[:, 1], "--", color=C_REF, lw=2.2, label="เส้นทางที่สั่ง", zorder=3)
        first = True
        for stamp in m[key]["runs"]:
            d = load_run(stamp)
            sel = d["valid"] & (d["phase"] == 3)
            ax.plot(d["x_m"][sel] * 1000, d["y_m"][sel] * 1000, color=C_REAL, lw=0.6, alpha=0.35,
                    label=f"ลูกบอล (จริง {len(m[key]['runs'])} รอบ)" if first else None)
            first = False
        ax.set_title(m[key]["label"], fontsize=10)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect("equal")
        ax.set_xlim(-42, 42)
        ax.set_ylim(-42, 42)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    return Figure("real_paths", "4.8", _save(fig, "fig4_08_real_paths.png"),
                  "ตำแหน่งลูกบอลขณะเดินตามเส้นทางวงกลมและหกเหลี่ยม บนแพลตฟอร์มจริง (ซ้อนทุกรอบ)",
                  "เส้นประส้มคือเส้นทางที่สั่ง เส้นแดงจางๆ คือตำแหน่งลูกบอลที่ตรวจจับได้ของแต่ละรอบ ซ้อนกัน "
                  "ยิ่งเข้มยิ่งเป็นย่านที่ลูกบอลอยู่บ่อย",
                  "ดูว่าเส้นแดงไปถึงเส้นประส้มหรือไม่ และรูปทรงของหกเหลี่ยม (มุม) ปรากฏในการเคลื่อนที่จริงหรือไม่")


# ----------------------------------------------------------------------------------------------
# 4.4 comparison
# ----------------------------------------------------------------------------------------------
def fig_compare_balance(m: dict) -> Figure:
    model = ST.Model(*balance_gains(m))
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    T, E = ST.step_response(model, 0.07, secs=30.0, delay_s=ST.DEFAULT_DELAY_S)
    first = True
    for stamp in m["balance"]["runs"]:
        t, r = balance_curve(stamp)
        if len(t) < 300:
            continue
        ax.plot(t, r / r[:5].mean() * 100.0, color=C_REAL, lw=0.9, alpha=0.45,
                label="ผลจริง (แต่ละรอบ)" if first else None)
        first = False
    ax.plot(T, np.abs(E) / 0.07 * 100.0, color=C_SIM, lw=3.0, label="จำลอง")
    ax.set_xlabel("เวลา (วินาที)")
    ax.set_ylabel("ระยะที่เหลือ (% ของระยะเริ่มต้น)")
    ax.set_title("เปรียบเทียบการเข้าสู่ศูนย์กลาง: จำลองกับผลจริง")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    return Figure("compare_balance", "4.12", _save(fig, "fig4_12_compare_balance.png"),
                  "ระยะที่เหลือถึงศูนย์กลาง คิดเป็นร้อยละของระยะเริ่มต้น เพื่อให้เทียบรอบที่เริ่มไม่เท่ากันได้",
                  "เส้นน้ำเงินหนาคือผลจำลอง เส้นแดงบางคือผลจริงแต่ละรอบ",
                  "ช่วงแรกทั้งจำลองและผลจริงลดลงในทิศทางและช่วงเวลาใกล้เคียงกัน (แนวโน้มเหมือนกัน) "
                  "แต่ช่วงหลังผลจำลองลงไปถึงศูนย์ ส่วนผลจริงแกว่งขึ้นลงและไม่ลงถึงศูนย์ (ต่างกัน)")


def fig_compare_radius(m: dict) -> Figure:
    g, k_i = balance_gains(m), chosen_k_i(m)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True)
    for ax, shape in zip(axes, ("circle", "hexagon")):
        T, P, R = sim_path(shape, g, k_i)
        ax.plot(T, np.hypot(*R.T) * 1000, "--", color=C_REF, lw=2.4, label="ระยะเส้นทางที่สั่งจากศูนย์กลาง")
        first = True
        for stamp in m[shape]["runs"]:
            d = load_run(stamp)
            v = d["valid"]
            ax.plot(d["t_s"][v] - d["t0"], np.hypot(d["x_m"][v], d["y_m"][v]) * 1000, color=C_REAL, lw=0.7,
                    alpha=0.35, label="ผลจริง (แต่ละรอบ)" if first else None)
            first = False
        ax.plot(T, np.hypot(*P.T) * 1000, color=C_SIM, lw=2.6, label="จำลอง")
        ax.set_title(LABELS[shape], fontsize=10)
        ax.set_xlabel("เวลา (วินาที)")
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 45)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("ระยะของลูกบอลจากศูนย์กลาง (mm)")
    axes[0].legend(fontsize=8, loc="upper left")
    return Figure("compare_radius", "4.13", _save(fig, "fig4_13_compare_radius.png"),
                  "ระยะของลูกบอลจากศูนย์กลางขณะเดินตามเส้นทาง เทียบกับระยะที่สั่ง (30–32 mm)",
                  "ถ้าลูกบอลเดินตามเส้นทางได้ครบ เส้นน้ำเงินและเส้นแดงต้องอยู่ระดับเดียวกับเส้นประส้ม",
                  "ผลจำลองไปถึงระยะที่สั่ง แต่ผลจริงอยู่ต่ำกว่ามากในทั้งสองรูปทรง (ต่างกันชัดเจน) "
                  "ส่วนจังหวะการวนยังสอดคล้องกับเส้นทาง")


def open_loop_points(m: dict) -> dict[str, list[tuple[float, float]]]:
    """(commanded tilt deg, ball travel mm in the first 0.8 s) for each axis, from the tilt sweeps."""
    out: dict[str, list[tuple[float, float]]] = {}
    for axis in ("x", "y"):
        pts = []
        for tilt, stamp in m["tilt_sweep"][axis]:
            with open(LOGS / f"tilt_test_{axis}_{tilt:g}deg_{stamp}.csv", newline="", encoding="utf-8") as h:
                rows = list(csv.DictReader(h))
            t = np.array([float(r["t_s"]) for r in rows])
            tr = np.array([r["tracked"] == "1" for r in rows])
            pos = np.array([float(r[f"{axis}_cm"]) for r in rows]) * 10.0
            phase = np.array([r["phase"] for r in rows])
            base = float(np.median(pos[(phase == "baseline") & tr][-10:]))
            t_plus = t[phase == "tilt+"][0]
            win = tr & (t >= t_plus + 0.15) & (t <= t_plus + 0.95)
            sign = 1.0 if axis == "x" else -1.0
            d = sign * (pos[win] - base)
            pts.append((float(tilt), float(d[np.argmax(np.abs(d))])))
        out[axis] = pts
    return out


def theory_open_loop_mm_per_rad() -> float:
    """Ball travel in the same 0.15-0.95 s window per radian of tilt, from the design model."""
    tg = np.arange(0.0, 3.0, 0.002)
    resp = _lagged_double_integral(tg, np.ones_like(tg), TA, ST.DEFAULT_DELAY_S) * ST.G_EFF * 1000.0
    win = (tg >= 0.15) & (tg <= 0.95)
    return float(np.max(np.abs(resp[win])))


def fig_open_loop(m: dict) -> Figure:
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    pts = open_loop_points(m)
    for axis, colour, mark in (("x", C_REAL, "o"), ("y", "#9467bd", "s")):
        ax.plot([0] + [p[0] for p in pts[axis]], [0] + [p[1] for p in pts[axis]], marker=mark, color=colour,
                lw=1.8, label=f"ผลจริง แกน {axis}")
    xs = np.array([0.0, 2.2])
    ax.plot(xs, theory_open_loop_mm_per_rad() * np.radians(xs), color=C_SIM, ls="--", lw=2.4,
            label="ทฤษฎี (แบบจำลองเชิงเส้น)")
    ax.axvspan(0, 0.5, color=C_GREY, alpha=0.15)
    ax.text(0.25, 52, "ช่วงที่ลูกบอล\nไม่ขยับ", ha="center", fontsize=9, color="#444444")
    ax.set_xlabel("มุมเอียงที่สั่ง (องศา)")
    ax.set_ylabel("ระยะที่ลูกบอลกลิ้งใน 0.8 วินาที (mm)")
    ax.set_title("ทดสอบวงเปิด: มุมที่สั่งเทียบกับระยะที่ลูกบอลกลิ้งได้")
    ax.set_xlim(0, 2.2)
    ax.set_ylim(-3, 62)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    return Figure("open_loop", "4.14", _save(fig, "fig4_14_open_loop.png"),
                  "ระยะที่ลูกบอลกลิ้งได้เมื่อเอียงแผ่นด้วยมุมต่างกัน โดยไม่ผ่านตัวควบคุม",
                  "เส้นประน้ำเงินคือที่ทฤษฎีทำนาย เส้นแดงและม่วงคือที่วัดได้จริงบนแกน x และ y",
                  "ที่มุมเอียงใหญ่ (1–2°) ผลจริงใกล้ทฤษฎี แต่ที่มุมเล็กกว่าประมาณ 0.5° ลูกบอลแทบไม่ขยับเลย "
                  "ทฤษฎีแบบเส้นตรงไม่มีช่วงนี้ นี่คือสาเหตุหลักที่ผลจริงต่างจากผลจำลองเมื่อคำสั่งเล็ก")


# ----------------------------------------------------------------------------------------------
# 4.3 path following: transient and steady state shown separately, mean +- 1 SD, cross-track error
# ----------------------------------------------------------------------------------------------
def _lost_flags(m: dict, key: str) -> list[bool]:
    zone = RM.default_zone_cm()
    return [bool(RM.score_run(LOGS / f"run_{s}.csv", zone).get("lost")) for s in m[key]["runs"]]


def fig_transient(m: dict) -> Figure:
    import path_analysis as PA
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    for ax, key in zip(axes, ("circle", "hexagon")):
        grid, mu, sd, arr = PA.transient_curves(m[key]["runs"])
        for row in arr:
            ax.plot(grid, row, color=C_GREY, lw=0.6, alpha=0.5)
        ax.fill_between(grid, np.maximum(mu - sd, 0), mu + sd, color=C_REAL, alpha=0.25, label="ผลจริง ±1 SD")
        ax.plot(grid, mu, color=C_REAL, lw=2.2, label=f"ผลจริง ค่าเฉลี่ย {len(arr)} รอบ")
        ax.axhline(40.0, color=C_GREY, ls=":", lw=1.0)
        ax.axvline(HOLD_S, color="black", ls="--", lw=0.8)
        ax.text(HOLD_S - 0.1, 94, "เริ่มขยายเส้นทาง", fontsize=8, ha="right")
        ax.set_title(LABELS[key], fontsize=10)
        ax.set_xlabel("เวลานับจากตรวจพบลูกบอลครั้งแรก (วินาที)")
        ax.set_xlim(0, 11)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.68))
    axes[0].set_ylabel("ระยะห่างจากจุดเป้าหมาย (mm)")
    return Figure("transient", "4.9", _save(fig, "fig4_09_transient.png"),
                  "ช่วงวิ่งเข้าหาเป้าหมาย (transient): 6 วินาทีแรกทรงตัวที่ศูนย์กลาง และ 5 วินาทีที่เส้นทางค่อยๆ ขยาย",
                  "เส้นเทาบางคือแต่ละรอบ เส้นแดงคือค่าเฉลี่ย แถบแรเงาคือ ±1 SD เส้นประดำคือจุดที่เส้นทางเริ่มขยาย เส้นประเทาคือโซนศูนย์กลาง 4 cm",
                  "ลูกบอลที่วางอิสระเข้าสู่โซนศูนย์กลางภายในไม่กี่วินาที ช่วงนี้แยกออกจากช่วงเกาะเส้นทางเพื่อไม่ให้จุดเริ่มที่ต่างกันมารบกวนการเทียบกับทฤษฎี")


def fig_polar_band(m: dict) -> Figure:
    import path_analysis as PA
    g, k_i = balance_gains(m), chosen_k_i(m)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True)
    for ax, key in zip(axes, ("circle", "hexagon")):
        ang, mu, sd, n = PA.polar_band(m[key]["runs"], key)
        ax.plot(ang, PA.reference_radius_mm(key, ang), color=C_REF, lw=2.6, label="เส้นทางที่สั่ง (ทฤษฎี)")
        Tm, Pm, _R = sim_path(key, g, k_i)
        sel = Tm > 40.0      # last laps only, after the start-up transient of the simulation
        sang = np.degrees(np.arctan2(Pm[sel, 1], Pm[sel, 0]))
        srad = np.hypot(Pm[sel, 0], Pm[sel, 1]) * 1000
        order = np.argsort(sang)
        ax.plot(sang[order], srad[order], color=C_SIM, lw=1.6, ls="--", label="จำลอง")
        ax.fill_between(ang, np.maximum(mu - sd, 0), mu + sd, color=C_REAL, alpha=0.25, label="ผลจริง ±1 SD ระหว่างรอบ")
        ax.plot(ang, mu, color=C_REAL, lw=2.2, label=f"ผลจริง ค่าเฉลี่ย {len(m[key]['runs'])} รอบ")
        ax.set_title(LABELS[key], fontsize=10)
        ax.set_xlabel("ทิศของลูกบอลรอบศูนย์กลาง (องศา)")
        ax.set_xlim(-180, 180)
        ax.set_ylim(0, 45)
        ax.set_xticks(range(-180, 181, 60))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("ระยะของลูกบอลจากศูนย์กลาง (mm)")
    axes[0].legend(fontsize=8, loc="lower left")
    return Figure("polar_band", "4.10", _save(fig, "fig4_10_polar_band.png"),
                  "ช่วงเกาะเส้นทาง (steady state): ระยะจากศูนย์กลางของลูกบอลตามทิศรอบศูนย์กลาง เทียบกับเส้นทางที่สั่ง",
                  "เส้นทึบส้มคือเส้นทางทฤษฎี เส้นประน้ำเงินคือผลจำลอง เส้นแดงคือค่าเฉลี่ยของผลจริง แถบแรเงาคือ ±1 SD ระหว่างรอบ "
                  "(ใช้เฉพาะช่วงที่ลูกบอลอยู่บนเส้นทางเต็มขนาด และไม่ขึ้นกับเวลา จึงไม่ถูกกระทบจากความล่าช้า)",
                  "ถ้าเส้นแดงอยู่ต่ำกว่าเส้นส้ม แปลว่าลูกบอลวนเป็นวงเล็กกว่าที่สั่ง แถบแรเงากว้างแปลว่ารอบต่างๆ ต่างกันมาก")


def fig_cross_track(m: dict) -> Figure:
    import path_analysis as PA
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    target = float(RM._limit("track_mm"))
    for ax, key in zip(axes, ("circle", "hexagon")):
        lost = _lost_flags(m, key)
        vals = [PA.run_summary(s, key) for s in m[key]["runs"]]
        cross = [v["cross_mm"] for v in vals]
        sync = [v["sync_mm"] for v in vals]
        x = np.arange(1, len(cross) + 1)
        ax.bar(x - 0.2, sync, 0.4, color="#cfcfcf", label="เทียบจุดอ้างอิง ณ เวลาเดียวกัน")
        bars = ax.bar(x + 0.2, cross, 0.4, color=C_REAL, label="ระยะสั้นสุดถึงเส้นทาง (cross-track)")
        for b, is_lost in zip(bars, lost):
            if is_lost:
                b.set_hatch("//")
                b.set_edgecolor("black")
        ax.axhline(np.mean(cross), color=C_REAL, ls="--", lw=1.4)
        ax.text(len(cross) + 0.6, np.mean(cross) + 0.6, f"เฉลี่ย {np.mean(cross):.1f} mm", color=C_REAL, fontsize=9, ha="right")
        ax.axhline(target, color="black", ls=":", lw=1.2)
        ax.text(0.6, target + 0.6, f"เป้าหมาย {target:g} mm", fontsize=9, bbox=dict(fc="white", ec="none", pad=1))
        ax.set_title(LABELS[key] + " (ลายขีด = ลูกบอลหลุดระหว่างรอบ)", fontsize=10)
        ax.set_xlabel("รอบที่")
        ax.set_xticks(x)
        ax.set_ylim(0, 45)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("ความคลาดเฉลี่ยของแต่ละรอบ (mm)")
    axes[0].legend(fontsize=8, loc="upper left")
    return Figure("cross_track", "4.11", _save(fig, "fig4_11_cross_track.png"),
                  "ความคลาดเคลื่อนเฉลี่ยของแต่ละรอบ ช่วงเกาะเส้นทาง: วัดเทียบเวลากับวัดเป็นระยะจากเส้นทาง",
                  "แท่งเทาวัดกับจุดที่ควรอยู่ ณ เวลาเดียวกัน (ลูกบอลที่วิ่งตามหลังจะถูกนับเป็นความคลาดมาก) "
                  "แท่งแดงวัดระยะสั้นสุดจากลูกบอลถึงเส้นทาง (ไม่นับความล่าช้า)",
                  "ความต่างระหว่างแท่งเทากับแท่งแดงคือส่วนที่เกิดจากความล่าช้า ไม่ใช่การหลุดออกจากเส้นทาง")


BUILDERS = (fig_poles, fig_margin_vs_delay, fig_root_locus, fig_sim_response, fig_sim_path,
            fig_real_balance, fig_start_vs_result, fig_real_paths, fig_transient, fig_polar_band,
            fig_cross_track, fig_compare_balance,
            fig_compare_radius, fig_open_loop)


def build_all() -> dict[str, Figure]:
    m = manifest()
    figures: dict[str, Figure] = {}
    for builder in BUILDERS:
        figure = builder(m)
        figures[figure.key] = figure
        print(f"  รูปที่ {figure.number}  {figure.path.name}")
    return figures


if __name__ == "__main__":
    print("building figures:")
    build_all()
