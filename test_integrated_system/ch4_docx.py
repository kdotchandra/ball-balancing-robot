"""Write chapters 4-5 of the project report as a Word file, following the report outline.

    python ch4_docx.py                        # -> logs/chapter4_5_report.docx
    python ch4_docx.py -o /path/name.docx

Outline (as required by the report template):
    4   ผลการดำเนินโครงงาน
    4.1 ผลการวิเคราะห์เสถียรภาพของระบบควบคุม      (poles, Routh-Hurwitz, gain/phase margin, root locus, integral term)
    4.2 ผลการจำลองการทำงานของระบบด้วย Python       (response curve, overshoot, settling time, steady-state error)
    4.3 ผลการทดสอบสมรรถนะบนแพลตฟอร์มจริง
    4.4 ผลเปรียบเทียบระหว่างผลการจำลองและผลการทดลองจริง
    5   สรุปผลการดำเนินโครงงาน  (5.1 สรุปผล, 5.2 ข้อเสนอแนะ)

Every number in the document is computed when it is built: the simulation from stability.py, the
real-platform numbers from the counted trials of the tags balance / circle / hexagon (sidecar files
written by main.py, scored by run_metrics.py), and the figures from ch4_figures.py. Run it again after
new trials; nothing is typed in by hand. The body font is TH SarabunPSK (change BODY_FONT if the report
template uses another).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

import ch4_figures as CF
import experiment_profile as PROF
import run_metrics as RM
import stability as ST
import theory_limits as TH
from params import TA

BODY_FONT = "TH SarabunPSK"
BODY = Pt(15)
SMALL = Pt(13)
SMALL1 = Pt(14)
SMALLER = Pt(12)
TEXT_WIDTH_IN = 5.95      # A4 minus margins
OUT_DEFAULT = CF.LOGS / "chapter4_5_report.docx"

# Fixed table numbers (so text can refer to a table before it appears)
T = {"params": "4.1", "poles": "4.2", "routh": "4.3", "margins": "4.4", "lqi": "4.5", "sim": "4.6",
     "conditions": "4.7", "balance": "4.8", "balance_stats": "4.9", "balance_bins": "4.10", "paths": "4.11",
     "cross": "4.12", "ladder": "4.13", "supp": "4.14", "compare": "4.15", "tuning": "4.16"}


# ----------------------------------------------------------------------------------------------
# document helpers
# ----------------------------------------------------------------------------------------------
def _style_run(run, size=BODY, bold=False, italic=False) -> None:
    run.font.name = BODY_FONT
    run.font.size = size
    run.bold = bold
    run.italic = italic
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), BODY_FONT)
    szcs = OxmlElement("w:szCs")
    szcs.set(qn("w:val"), str(int(round(float(size) / 12700.0 * 2))))
    rpr.append(szcs)


def para(doc, text="", size=BODY, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=6, indent=None):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(after)
    if indent is not None:
        p.paragraph_format.first_line_indent = Inches(indent)
    _style_run(p.add_run(text), size=size, bold=bold, italic=italic)
    return p


def heading(doc, text, level=1):
    sizes = {0: Pt(22), 1: Pt(19), 2: Pt(17), 3: Pt(15)}
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt({0: 0, 1: 18, 2: 12, 3: 8}[level])
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    _style_run(p.add_run(text), size=sizes[level], bold=True)
    return p


def bullets(doc, items, size=BODY):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(3)
        _style_run(p.add_run(item), size=size)


def _shade(cell, fill: str) -> None:
    tc_pr = cell._element.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def callout(doc, title: str, lines: list[str], fill: str = "EAF2FB") -> None:
    """A shaded box: the short takeaway a reader can stop at."""
    t = doc.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = t.rows[0].cells[0]
    _shade(cell, fill)
    first = cell.paragraphs[0]
    first.paragraph_format.space_after = Pt(3)
    _style_run(first.add_run(title), size=BODY, bold=True)
    for line in lines:
        p = cell.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(3)
        _style_run(p.add_run("• " + line), size=SMALL1)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def table(doc, key: str, caption: str, header, rows, widths=None, note: str | None = None) -> None:
    cap = para(doc, f"ตารางที่ {T[key]} {caption}", size=SMALL1, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, after=3)
    cap.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, text in zip(t.rows[0].cells, header):
        _shade(cell, "D9E2F3")
        cell.paragraphs[0].paragraph_format.space_after = Pt(2)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        _style_run(cell.paragraphs[0].add_run(str(text)), size=SMALL, bold=True)
    for row in rows:
        cells = t.add_row().cells
        for i, (cell, text) in enumerate(zip(cells, row)):
            cell.paragraphs[0].paragraph_format.space_after = Pt(2)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER
            _style_run(cell.paragraphs[0].add_run(str(text)), size=SMALL)
    if widths:
        scale = min(1.0, TEXT_WIDTH_IN / sum(widths))
        for row in t.rows:
            for cell, width in zip(row.cells, widths):
                cell.width = Inches(width * scale)
    if note:
        para(doc, note, size=SMALLER, italic=True, after=6)
    else:
        doc.add_paragraph().paragraph_format.space_after = Pt(2)


def figure(doc, fig: CF.Figure, width: float = 5.7) -> None:
    doc.add_picture(str(fig.path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.keep_with_next = True
    para(doc, f"รูปที่ {fig.number} {fig.title}", size=SMALL1, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    para(doc, fig.caption, size=SMALL, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=3)
    p = para(doc, "", size=SMALL, after=8)
    _style_run(p.add_run("อ่านกราฟนี้อย่างไร: "), size=SMALL, bold=True)
    _style_run(p.add_run(fig.reading), size=SMALL)


def fmt(value, digits=1, unit="", dash="-") -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return dash
    return f"{value:.{digits}f}{unit}"


def mean_sd(vals: list[float], digits: int = 1) -> str:
    vals = [v for v in vals if v == v]
    if not vals:
        return "-"
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return f"{np.mean(vals):.{digits}f} ± {sd:.{digits}f}"


def span(vals: list[float], digits: int = 1) -> str:
    vals = [v for v in vals if v == v]
    return f"{min(vals):.{digits}f}–{max(vals):.{digits}f}" if vals else "-"


def rate_text(k: int, n: int) -> str:
    lo, hi = RM.wilson_interval(k, n)
    return f"{k}/{n} ({100 * k / n:.0f}%, ช่วงเชื่อมั่น 95%: {100 * lo:.0f}–{100 * hi:.0f}%)" if n else "-"


# ----------------------------------------------------------------------------------------------
# numbers used in the text
# ----------------------------------------------------------------------------------------------
def scored(m: dict, key: str) -> list[dict]:
    zone = RM.default_zone_cm()
    return [RM.score_run(CF.LOGS / f"run_{s}.csv", zone) | {"stamp": s} for s in m.get(key, {}).get("runs", [])]


def sim_zone_settle(model: ST.Model, start_mm: float, delay_s: float, zone_mm: float = 40.0) -> float:
    """Time after which the simulated ball stays inside the centre zone (same definition as the logs)."""
    Tm, E = ST.step_response(model, start_mm / 1000.0, secs=20.0, delay_s=delay_s)
    outside = np.where(np.abs(E) * 1000.0 > zone_mm)[0]
    return float(Tm[outside[-1]]) if len(outside) else 0.0


def command_noise_stats(m: dict) -> dict:
    """How noisy the velocity estimate and the tilt command are while following the path."""
    import csv
    from params import K_AXIS
    ks, kvs = m["balance"]["gains"]
    k1, k2 = K_AXIS[0] * ks, K_AXIS[1] * ks * kvs
    out = {"vel_std": [], "hi_frac": [], "rev": [], "vel_term": [], "pos_term": []}
    for stamp in m["circle"]["runs"]:
        with open(CF.LOGS / f"run_{stamp}.csv", newline="", encoding="utf-8") as h:
            rows = list(csv.DictReader(h))
        col = lambda c: np.array([float(r[c]) if r[c] != "" else np.nan for r in rows])
        ok = (col("control_valid") == 1) & (col("path_phase") == 3)
        if ok.sum() < 100:
            continue
        t, v = col("t_s")[ok], col("x_dot_mps")[ok]
        e, cmd = (col("x_m") - col("x_ref_m"))[ok], col("theta_x_cmd_deg")[ok]
        dt = float(np.median(np.diff(t)))
        spec = np.abs(np.fft.rfft(v - v.mean())) ** 2
        freq = np.fft.rfftfreq(len(v), dt)
        d = np.diff(cmd)
        out["vel_std"].append(float(np.std(v) * 1000.0))
        out["hi_frac"].append(float(spec[freq >= 0.7].sum() / spec.sum() * 100.0))
        out["rev"].append(float(np.sum(np.diff(np.sign(d[np.abs(d) > 0.02])) != 0) / (t[-1] - t[0])))
        out["vel_term"].append(float(np.degrees(k2 * v).std()))
        out["pos_term"].append(float(np.degrees(k1 * e).std()))
    return {k: float(np.mean(v)) for k, v in out.items()}


def apply_measured_delay(m: dict) -> dict:
    """If test_freq_response.py logs are listed in report_runs.json, use the measured loop delay."""
    import test_freq_response as FR
    results = []
    for name in m.get("freq_response", []):
        path = CF.LOGS / (name if name.endswith(".csv") else name + ".csv")
        results.append((path.name, FR.analyze_file(path)))
    summary = FR.summarize([r for _n, r in results]) if results else {"ok": False}
    if not summary.get("ok"):
        return {"measured": False, "delay": ST.DEFAULT_DELAY_S, "rows": results}
    ST.DEFAULT_DELAY_S = summary["delay_s"]
    ST.DELAY_RANGE_S = (max(0.0, summary["delay_min"]), max(summary["delay_max"], summary["delay_min"] + 0.02))
    return {"measured": True, "delay": summary["delay_s"], "min": summary["delay_min"], "max": summary["delay_max"],
            "g_eff": summary["g_eff"], "rows": results}


# ----------------------------------------------------------------------------------------------
# the document
# ----------------------------------------------------------------------------------------------
def build(out_path: Path) -> Path:
    m = CF.manifest()
    delay_info = apply_measured_delay(m)
    is_measured = delay_info["measured"]
    delay_word = "วัดได้จากการทดสอบวงเปิด" if is_measured else "ประมาณการ"
    figs = CF.build_all()
    g = CF.balance_gains(m)
    k_i = CF.chosen_k_i(m)
    bal = ST.Model(*g)
    d_est = ST.DEFAULT_DELAY_S
    g_eff = ST.G_EFF
    provisional = [k for k in ("balance", "circle", "hexagon") if not m[k].get("from_tags")]

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = BODY_FONT
    style.font.size = BODY
    style.element.rPr.rFonts.set(qn("w:cs"), BODY_FONT)
    for section in doc.sections:
        section.page_width, section.page_height = Inches(8.27), Inches(11.69)      # A4
        section.top_margin = section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.25)
        section.right_margin = Inches(1.0)

    # =========================================================== 4
    heading(doc, "บทที่ 4 ผลการดำเนินโครงงาน", 0)
    para(doc, "บทนี้รายงานผลของระบบควบคุมการทรงตัวของลูกบอลบนแผ่นกลไก 3RRS เรียงตามลำดับ ได้แก่ "
              "การวิเคราะห์เสถียรภาพทางทฤษฎี (4.1) ผลการจำลองด้วย Python (4.2) ผลการทดสอบบนแพลตฟอร์มจริง (4.3) "
              "และการเปรียบเทียบผลจำลองกับผลจริงเพื่อดูว่าแนวโน้มเหมือนหรือต่างกันอย่างไรและเพราะอะไร (4.4)", indent=0.4)
    para(doc, "ระบบทำงานดังนี้ กล้องตรวจจับตำแหน่งลูกบอลบนแผ่น ตัวควบคุมคำนวณมุมเอียงที่ต้องการของแผ่นจากตำแหน่ง "
              "ความเร็วของลูกบอล และมุมเอียงปัจจุบัน แล้วส่งคำสั่งให้เซอร์โวสามตัวเอียงแผ่น การวิเคราะห์ในบทนี้ทำแยกทีละแกน "
              "(แกน x และ y เหมือนกัน) โดยใช้แบบจำลองเดียวกับที่ใช้ออกแบบตัวควบคุมแบบ LQR", indent=0.4)
    para(doc, f"การทดสอบทุกรายการใช้ตัวควบคุมชุดเดียวกัน (เกนตำแหน่ง k = {g[0]:g}, เกนความเร็ว kv = {g[1]:g}) "
              "เพื่อให้เปรียบเทียบผลระหว่างการทดสอบได้ ทดสอบสามรายการ ได้แก่ การรักษาลูกบอลให้อยู่ใกล้จุดศูนย์กลาง "
              "การติดตามเส้นทางวงกลมรัศมี 3 cm และการติดตามเส้นทางหกเหลี่ยมที่ครอบวงกลมรัศมี 3 cm รายการละ 10 ครั้ง",
         indent=0.4)
    if provisional:
        para(doc, "หมายเหตุ: ข้อมูลของการทดลอง " + ", ".join(CF.LABELS[k] for k in provisional) +
                  " ในฉบับนี้ยังเป็นข้อมูลชั่วคราวจากการทดลองก่อนหน้า (ยังไม่ครบ 10 ครั้งตามแผน) จะเปลี่ยนเป็นข้อมูลจริงเมื่อสร้างเอกสารใหม่หลังทดลอง",
             size=SMALL, italic=True)

    # ------------------------------------------------------- 4.1
    heading(doc, "4.1 ผลการวิเคราะห์เสถียรภาพของระบบควบคุม", 1)
    heading(doc, "4.1.1 แบบจำลองและพารามิเตอร์ที่ใช้", 2)
    para(doc, "แบบจำลองต่อหนึ่งแกนมีสามตัวแปรสถานะ ได้แก่ ระยะคลาดของลูกบอลจากเป้าหมาย (e) ความเร็วคลาด (v) และมุมเอียงของแผ่น (θ) "
              "ลูกบอลกลิ้งบนแผ่นเอียงด้วยความเร่ง a = g_eff · θ และแผ่นตอบสนองต่อคำสั่งแบบหน่วงอันดับหนึ่งด้วยค่าคงที่เวลา TA "
              "ตัวควบคุมคำนวณคำสั่งมุมเอียงเป็น u = −(k1·e + k2·v + k3·θ)", indent=0.4)
    para(doc, "เกน K ได้จากการออกแบบ LQR ที่ลดฟังก์ชันต้นทุน J = ∫(xᵀQx + uᵀRu)dt โดยให้น้ำหนักกับระยะคลาดของลูกบอล (e) มากที่สุด "
              "เพราะเป็นตัวแปรที่ต้องการควบคุม ให้น้ำหนักกับความเร็วคลาด (v) และมุมเอียง (θ) รองลงมาเพื่อกดการแกว่งและไม่ให้แผ่นเอียงเร็วเกิน "
              "และให้ R เป็นน้ำหนักของคำสั่งมุมเอียง", indent=0.4)
    para(doc, "[ต้องแก้ก่อนส่ง: ใส่ค่าเมทริกซ์ Q และ R ที่ใช้จริงจากโน้ตบุ๊กออกแบบ LQR ของโครงงาน (calculated_lqr_ik_3rrs.ipynb) "
              "โปรแกรมนี้มีเฉพาะค่าเกน K ที่ได้ จึงไม่สามารถใส่ค่า Q และ R ให้เองได้]", size=SMALL, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    table(doc, "params", "พารามิเตอร์ของแบบจำลองและเกนของตัวควบคุมที่ใช้ในการทดลอง",
          ["พารามิเตอร์", "ค่า", "ที่มา"],
          [["อัตราขยายของลูกบอล g_eff", f"{g_eff:.3f} m/s² ต่อ rad", "ลูกทรงกลมกลวงกลิ้งบนพื้นเอียง a = (3/5)·g·sinθ"],
           ["ค่าคงที่เวลาของแผ่น TA", f"{TA:.3f} วินาที", "จากพารามิเตอร์ที่ใช้ออกแบบ LQR"],
           ["เกนของตัวควบคุม (k1, k2, k3)", ", ".join(f"{v:.3f}" for v in bal.K[-3:]),
            f"K ของ LQR × [{g[0]:g}, {g[0]:g}×{g[1]:g}, {g[0]:g}]"],
           ["เวลาหน่วงเพิ่มในลูป", f"{d_est:.2f} วินาที ({delay_word})",
            (f"วัดจากการเอียงแผ่นแบบไซน์หลายความถี่ ช่วง {ST.DELAY_RANGE_S[0]:.2f}–{ST.DELAY_RANGE_S[1]:.2f} วินาที"
             if is_measured else
             f"กล้องและการประมวลผลภาพประมาณ 3 เฟรม รวมคาบส่งคำสั่ง ช่วงที่คาดว่าเป็น {ST.DELAY_RANGE_S[0]:g}–{ST.DELAY_RANGE_S[1]:g} วินาที")]],
          widths=[2.0, 1.6, 2.6],
          note=("หมายเหตุ เวลาหน่วงวัดจากการทดสอบวงเปิด" if is_measured else
                "หมายเหตุ เวลาหน่วงเป็นค่าที่ประมาณ ยังไม่ได้วัดตรง จึงวิเคราะห์เป็นช่วงในหัวข้อ 4.1.4"))

    heading(doc, "4.1.2 ตำแหน่งโพลของระบบวงปิด", 2)
    p = ST.poles(bal)
    pair = max((x for x in p if x.imag > 0), key=lambda x: x.imag)
    real_pole = [x.real for x in p if abs(x.imag) < 1e-9][0]
    table(doc, "poles", "โพลของระบบวงปิด (ยังไม่รวมเวลาหน่วง)",
          ["โพลคู่เชิงซ้อน", "โพลจริง", "ความถี่ธรรมชาติ (rad/s)", "อัตราการหน่วง ζ"],
          [[f"{pair.real:.2f} ± {pair.imag:.2f}j", f"{real_pole:.2f}", f"{abs(pair):.2f}", f"{-pair.real / abs(pair):.2f}"]],
          widths=[1.6, 1.0, 1.9, 1.5])
    figure(doc, figs["poles"])
    para(doc, "ผลคือโพลทุกตัวมีส่วนจริงเป็นลบ ระบบจึงเสถียร โพลคู่ที่มีส่วนจินตภาพเป็นตัวกำหนดการแกว่งของการตอบสนอง", indent=0.4)

    heading(doc, "4.1.3 เกณฑ์ Routh-Hurwitz", 2)
    para(doc, "สมการลักษณะเฉพาะของระบบวงปิดเป็นอันดับสาม s³ + a₂s² + a₁s + a₀ = 0 ระบบเสถียรก็ต่อเมื่อสัมประสิทธิ์ทุกตัวเป็นบวก "
              "และ a₂·a₁ > a₀ (คอลัมน์แรกของตาราง Routh ไม่มีการเปลี่ยนเครื่องหมาย)", indent=0.4)
    r = ST.routh_table(bal)
    table(doc, "routh", "ผลการตรวจด้วยเกณฑ์ Routh-Hurwitz (ยังไม่รวมเวลาหน่วง)",
          ["a₂", "a₁", "a₀", "a₂·a₁ − a₀ (ต้องมากกว่า 0)", "ผล"],
          [[f"{r['a2']:.2f}", f"{r['a1']:.2f}", f"{r['a0']:.2f}", f"{r['condition']:.1f}", "เสถียร" if r["stable"] else "ไม่เสถียร"]],
          widths=[0.9, 0.9, 0.9, 2.3, 1.0])

    heading(doc, "4.1.4 Gain margin และ Phase margin", 2)
    para(doc, "Phase margin (PM) และ gain margin (GM) บอกว่าระบบมีที่เผื่อเท่าไรก่อนจะไม่เสถียร ค่ามากหมายถึงเผื่อมาก "
              "โดยทั่วไปถือว่า PM ควรมากกว่าประมาณ 30° เวลาหน่วงในลูป (จากกล้องและการประมวลผลภาพ) ลดที่เผื่อนี้ลงโดยตรง "
              + ("ตารางจึงแสดงผลที่เวลาหน่วงหลายค่า โดยแถวที่ระบุคือค่าที่วัดได้" if is_measured else
                 "และเป็นค่าที่ยังไม่ทราบแน่ชัด ตารางจึงแสดงผลที่เวลาหน่วงหลายค่า"), indent=0.4)
    rows = []
    for delay in sorted({0.0, 0.10, round(d_est, 3), 0.20, 0.25}):
        mg = ST.margins(bal, delay)
        rows.append([f"{delay:g}" + (f" ({delay_word})" if abs(delay - d_est) < 1e-9 else ""),
                     f"{mg['pm_deg']:.0f}°", fmt(mg["gm_db"], 0, " dB", "ไม่จำกัด")])
    table(doc, "margins", "Phase margin และ Gain margin ที่เวลาหน่วงต่างๆ",
          ["เวลาหน่วง (s)", "Phase margin", "Gain margin"], rows, widths=[2.4, 1.5, 1.5])
    if is_measured:
        rows = [[n.replace("freq_", "").replace(".csv", ""), f"{r['freq_hz']:g}", f"{r['amp_mm']:.1f}", f"{r['g_eff']:.2f}", f"{r['delay_s']:.3f}"]
                for n, r in delay_info["rows"] if r.get("ok")]
        para(doc, f"เวลาหน่วงที่วัดได้: ค่ากลาง {delay_info['delay']:.3f} วินาที (ช่วง {delay_info['min']:.3f}–{delay_info['max']:.3f}) "
                  f"อัตราขยายที่วัดได้ {delay_info['g_eff']:.2f} m/s² ต่อ rad เทียบกับแบบจำลอง {g_eff:.3f}", indent=0.4)
    figure(doc, figs["margin_delay"])
    cd = ST.critical_delay(bal)
    para(doc, f"เวลาหน่วงที่ทำให้ระบบเริ่มไม่เสถียร (เวลาหน่วงวิกฤต) คือ {cd:.2f} วินาที ที่เวลาหน่วง {d_est:.2f} วินาที "
              f"ระบบยังเสถียรโดยมี PM {ST.margins(bal, d_est)['pm_deg']:.0f}°", indent=0.4)

    heading(doc, "4.1.5 Root locus", 2)
    figure(doc, figs["root_locus"])
    kc = ST.critical_gain_scale(g[1], d_est)
    para(doc, f"เมื่อเพิ่มเกนของตัวควบคุมโดยรักษาอัตราส่วนเกนความเร็วต่อเกนตำแหน่งไว้ (เวลาหน่วง {d_est:.2f} วินาที) "
              f"ระบบเริ่มไม่เสถียรที่ตัวคูณเกน k ≈ {kc:.2f} เทียบกับที่ใช้จริง k = {g[0]:g} (ประมาณ {kc / g[0]:.1f} เท่า)", indent=0.4)

    heading(doc, "4.1.6 ตัวอินทิเกรตสำหรับการเดินตามเส้นทาง (LQI)", 2)
    w_lap = 2.0 * math.pi / PROF.PATH_PERIOD_S
    para(doc, "การเดินตามเส้นทางเป็นการตามเป้าหมายที่เคลื่อนที่ ตัวควบคุมจึงใช้ความคลาดของตำแหน่งและความเร็วเทียบกับเส้นทาง "
              "(รูปแบบ error-state) อยู่แล้ว เพื่อให้แผ่นข้ามช่วงที่ลูกบอลไม่ขยับ (แรงฝืด) ได้ จึงเพิ่มตัวอินทิเกรตของความคลาดตำแหน่ง "
              "u = −(k_i·∫e dt + k1·e + k2·v + k3·θ) เฉพาะช่วงที่เส้นทางเคลื่อนที่ ส่วนเกนอีกสามตัวเป็นค่าเดิมทั้งหมด "
              "ตารางที่ 4.5 แสดงว่าการเพิ่มตัวอินทิเกรตเป็นอย่างไรในแบบจำลองเชิงเส้น", indent=0.4)
    rows = []
    for ki in (0.0, 0.05, 0.10, 0.20, 0.30, 0.50):
        mod = ST.Model(g[0], g[1], ki)
        k1 = mod.K[-3]
        rows.append([f"{ki:g}", f"{ST.margins(mod, 0.13)['pm_deg']:.0f}° / {ST.margins(mod, 0.20)['pm_deg']:.0f}°",
                     f"{ST.critical_delay(mod):.2f}", f"×{abs(k1 + ki / (1j * w_lap)) / k1:.2f}",
                     f"{ST.tracking_error_amplitude(mod, 0.030, PROF.PATH_PERIOD_S) * 1000:.2f}"])
    table(doc, "lqi", "ผลของตัวอินทิเกรต k_i ต่อความเสถียรและการติดตามในแบบจำลองเชิงเส้น",
          ["k_i (rad/(m·s))", "PM ที่ 0.13 s / 0.20 s", "เวลาหน่วงวิกฤต (s)", "เกนตำแหน่งเทียบเท่าที่ความถี่ของเส้นทาง",
           "ความคลาดสูงสุดตามวงกลม 3 cm (mm)"], rows, widths=[1.1, 1.4, 1.2, 1.6, 1.3],
          note=f"หมายเหตุ ตัวอินทิเกรตขนาดเล็กเกือบไม่เสียที่เผื่อเสถียรภาพ แต่ที่ k_i = 0.5 ที่เผื่อลดลงมาก จึงจำกัด k_i ไม่เกิน {PROF.PATH_KI_MAX:g} "
               "การวิเคราะห์เชิงเส้นไม่ทำนายการแกว่งที่เกิดจากแรงฝืดร่วมกับตัวอินทิเกรต ค่า k_i ที่ใช้จึงต้องเลือกจากการทดลองจริง (หัวข้อ 4.3.5)")

    callout(doc, "สรุปผลการวิเคราะห์เสถียรภาพ (4.1)", [
        "ระบบเสถียร: โพลทุกตัวอยู่ครึ่งซ้าย และผ่านเกณฑ์ Routh-Hurwitz",
        f"ที่เวลาหน่วง {d_est:.2f} วินาที ระบบมี Phase margin {ST.margins(bal, d_est)['pm_deg']:.0f}° และเวลาหน่วงวิกฤต {cd:.2f} วินาที",
        ("เวลาหน่วงของลูปวัดได้จากการทดสอบวงเปิด" if is_measured else
         "เวลาหน่วงของลูปเป็นตัวกำหนดความเสถียรมากที่สุด แต่ยังไม่ได้วัดตรง จึงแสดงผลเป็นช่วง (แนะนำให้วัดตรงในงานต่อไป ข้อ 5.2.2)"),
        "ตัวอินทิเกรตสำหรับเส้นทางเสียที่เผื่อเสถียรภาพน้อยมากเมื่อ k_i ไม่เกิน 0.3",
    ])

    # ------------------------------------------------------- 4.2
    heading(doc, "4.2 ผลการจำลองการทำงานของระบบด้วย Python", 1)
    para(doc, "จำลองแบบจำลองเชิงเส้นในหัวข้อ 4.1 ด้วย Python (ไฟล์ stability.py) ที่เกนเดียวกับการทดลองจริง ปล่อยลูกบอลห่างจากศูนย์กลาง 60 mm "
              "แล้วดูการตอบสนองทั้งกรณีอุดมคติและกรณีมีเวลาหน่วง", indent=0.4)
    figure(doc, figs["sim_response"])
    sm0, sm1 = ST.step_metrics(bal), ST.step_metrics(bal, delay_s=d_est)
    table(doc, "sim", "ผลการจำลองการเข้าสู่ศูนย์กลาง (ปล่อยห่าง 60 mm)",
          ["ตัวชี้วัด", "ไม่มีเวลาหน่วง", f"มีเวลาหน่วง {d_est:.2f} s"],
          [["Overshoot (% ของระยะเริ่มต้น)", f"{sm0['overshoot_pct']:.1f}", f"{sm1['overshoot_pct']:.1f}"],
           ["Settling time เข้า ±2% (วินาที)", f"{sm0['settle_2pct_s']:.2f}", f"{sm1['settle_2pct_s']:.2f}"],
           ["เวลาเข้าโซนศูนย์กลาง 4 cm (วินาที)", f"{sim_zone_settle(bal, 60, 0.0):.2f}", f"{sim_zone_settle(bal, 60, d_est):.2f}"],
           ["Steady-state error (mm)", f"{sm0['steady_state_m'] * 1000:.2f}", f"{sm1['steady_state_m'] * 1000:.2f}"]],
          widths=[2.8, 1.6, 1.6],
          note="หมายเหตุ Settling time (±2%) กรณีมีเวลาหน่วงสั้นกว่าเพียงเพราะการแกว่งช่วงแรกตกอยู่ในแถบ ±2% เร็วกว่า "
               "ทั้งสองกรณีให้ค่าใกล้เคียงกัน (ประมาณ 3 วินาที)")
    figure(doc, figs["sim_path"], width=5.95)
    sim = {}
    for shape in ("circle", "hexagon"):
        Tm, Pm, Rm = CF.sim_path(shape, g, k_i)
        sel = Tm > CF.HOLD_S + CF.RAMP_S
        sim[shape] = {"track": float(np.hypot(*(Pm[sel] - Rm[sel]).T).mean() * 1000),
                      "ratio": float(np.hypot(*Pm[sel].T).mean() / np.hypot(*Rm[sel].T).mean())}
    para(doc, f"ในการเดินตามเส้นทางคาบ 20 วินาที แบบจำลองให้ความคลาดเฉลี่ย {sim['circle']['track']:.1f} mm สำหรับวงกลม และ "
              f"{sim['hexagon']['track']:.1f} mm สำหรับหกเหลี่ยม ลูกบอลวนที่ระยะ {sim['circle']['ratio'] * 100:.0f}% และ "
              f"{sim['hexagon']['ratio'] * 100:.0f}% ของระยะที่สั่ง แปลว่าตามเส้นทางได้เกือบสนิท", indent=0.4)
    callout(doc, "สรุปผลการจำลอง (4.2)", [
        f"เข้าสู่ศูนย์กลางแล้วนิ่ง: overshoot ประมาณ {sm0['overshoot_pct']:.0f}–{sm1['overshoot_pct']:.0f}% เข้าโซน 4 cm ใน {sim_zone_settle(bal, 60, d_est):.1f} วินาที steady-state error เป็นศูนย์",
        f"เดินตามวงกลมได้ความคลาดเฉลี่ย {sim['circle']['track']:.1f} mm และหกเหลี่ยม {sim['hexagon']['track']:.1f} mm",
        "ผลนี้เป็นค่าอุดมคติของแบบจำลองเชิงเส้น ไม่รวมแรงฝืด ระยะฟรี และสัญญาณรบกวนของเซนเซอร์",
    ])

    # ------------------------------------------------------- 4.3
    heading(doc, "4.3 ผลการทดสอบสมรรถนะของระบบควบคุมบนแพลตฟอร์มจริง", 1)
    heading(doc, "4.3.1 เงื่อนไขการทดลอง", 2)
    table(doc, "conditions", "เงื่อนไขการทดลองบนแพลตฟอร์มจริง (เหมือนกันทุกการทดสอบ)",
          ["รายการ", "ค่า"],
          [["ลูกบอล", "ลูกปิงปอง เส้นผ่านศูนย์กลาง 40 mm"],
           ["การวัดตำแหน่ง", "กล้อง 30 เฟรมต่อวินาที ความละเอียด 1 px = 0.25 mm สัญญาณรบกวนประมาณ 0.75 mm"],
           ["เกนของตัวควบคุม", f"k = {g[0]:g}, kv = {g[1]:g} (ชุดเดียวกันทุกการทดสอบ)"],
           ["ตัวอินทิเกรตช่วงเส้นทาง", f"k_i = {k_i:g} rad/(m·s)" + (" (ปิด)" if k_i == 0 else "")],
           ["ส่งคำสั่งเซอร์โวทุก", f"{PROF.COMMAND_PERIOD_S:g} วินาที (เวลาเคลื่อนที่ต่อคำสั่ง {PROF.MOVE_MS} ms)"],
           ["ตัวจำกัดมุมเอียง", f"±{PROF.TILT_LIMIT_DEG:g}° และไม่เร็วเกิน {PROF.MAX_TILT_RATE_DEG_S:g}° ต่อวินาที"],
           ["การชดเชยความเอียงของแผ่น (trim)", f"เปิดใช้ (ค่าเรียนรู้ช้าๆ ไม่เกิน ±6°, k = {PROF.TRIM_KI:g})"],
           ["เส้นทาง", f"วงกลมรัศมี {PROF.CIRCLE_RADIUS_CM:g} cm และหกเหลี่ยมที่ครอบวงกลมรัศมี {PROF.HEXAGON_APOTHEM_CM:g} cm "
                       f"คาบ {PROF.PATH_PERIOD_S:g} วินาที เริ่มด้วยการทรงตัวที่ศูนย์กลาง {PROF.PATH_HOLD_S:g} วินาที แล้วขยายเส้นทางใน {PROF.PATH_RAMP_S:g} วินาที"],
           ["ความยาวการทดลอง", f"ทรงตัว {PROF.BALANCE_DURATION_S:g} วินาที เส้นทาง {PROF.PATH_DURATION_S:g} วินาที นับจากตรวจพบลูกบอลครั้งแรก"],
           ["การวางลูกบอลตอนเริ่ม (ทรงตัว)", "วางอิสระ ไม่กำหนดจุด ระบบบันทึกระยะและมุมที่เริ่มจริงจากข้อมูลทุกรอบ"],
           ["การนับผล", "นับทุกครั้งที่ระบบทำงานจนจบหรือลูกบอลหลุด ไม่นับครั้งที่ตรวจไม่พบลูกบอลเลย (ปัญหาการวางลูกบอล) และครั้งที่ผู้ทดลองหยุดเอง"]],
          widths=[2.2, 4.0])

    para(doc, "ข้อสังเกตเรื่องคุณภาพข้อมูล: ระหว่างการทดลองพบปัญหาของระบบตรวจจับ 2 เรื่องซึ่งแก้แล้วก่อนเก็บข้อมูลชุดสุดท้าย "
              "(1) เมื่อลูกบอลหายจากภาพชั่วคราว ความเร็วที่คำนวณหลังกลับมาเจอลูกบอลถูกหารด้วยเวลาผิด ทำให้ได้ความเร็วสูงเกินจริงและตัวควบคุมสั่งแผ่นเอียงกระตุก "
              "(2) ตัวตรวจจับบางครั้งล็อกไปที่หลอดไฟสว่างจ้านอกแผ่นแทนลูกบอล จึงกำหนดให้ตัวตรวจจับมองเฉพาะวัตถุที่อยู่ในรัศมี 11.5 cm จากศูนย์กลางแผ่น "
              "การทดลองที่ปนเปื้อนถูกเก็บแยกไว้ (ไม่ได้ลบ) และไม่นับในตาราง ข้อมูล 10 ครั้งของแต่ละรายการมาจากสองช่วงเวลาซึ่งใช้ตัวควบคุมและเกนชุดเดียวกัน "
              "ทุกรอบที่นับมีเฟรมที่ตรวจผิดที่หลอดไฟไม่เกิน 10 เฟรม นอกจากนี้ในบางรอบของการเดินตามเส้นทางลูกบอลตกจากแผ่น ผู้ทดลองใส่ลูกใหม่แทนระหว่างรอบ "
              "รอบเหล่านั้นนับเป็นรอบที่ลูกบอลหลุด (ล้มเหลว) และเป็นการแทรกแซงของผู้ทดลอง", indent=0.4)
    heading(doc, "4.3.2 ผลการทรงตัวที่จุดศูนย์กลาง", 2)
    para(doc, "วางลูกบอลไว้บนแผ่นอย่างอิสระแล้วปล่อยให้ระบบทรงตัวเอง บันทึกผลทุกรอบ ตัวชี้วัดคือเวลาที่เข้าโซนศูนย์กลาง 4 cm "
              "(อยู่ต่อเนื่อง 2 วินาที) RMS ของความคลาดใน 15 วินาทีสุดท้าย และความคลาดขณะนิ่งใน 5 วินาทีสุดท้าย", indent=0.4)
    figure(doc, figs["real_balance"])
    runs = scored(m, "balance")
    rows, ok_count = [], 0
    for i, r in enumerate(runs, start=1):
        start, angle = CF.start_info(r["stamp"])
        if r["kind"] == "short":
            rows.append([f"{i}", fmt(start), f"{angle:.0f}", "-", "-", "-", "-", "ไม่สำเร็จ (ลูกบอลหลุด)"])
            continue
        ok_count += 1 if r["success"] else 0
        rows.append([f"{i}", fmt(r["start_cm"]), f"{angle:.0f}", fmt(r["settle_s"]), fmt(r["rms_mm"]), fmt(r["steady_mm"]),
                     fmt(r["overshoot_pct"], 0), "สำเร็จ" if r["success"] else "ไม่สำเร็จ"])
    table(doc, "balance", "ผลการทรงตัวของแต่ละรอบ",
          ["รอบ", "ระยะเริ่ม (cm)", "ทิศเริ่ม (°)", "เข้าโซน 4 cm (s)", "RMS (mm)", "Steady-state (mm)", "Overshoot (%)", "ผล"],
          rows, widths=[0.4, 0.8, 0.7, 0.9, 0.7, 0.9, 0.8, 1.3],
          note="เกณฑ์สำเร็จ: ไม่สูญเสียลูกบอล และเข้าโซนศูนย์กลาง 4 cm อยู่ต่อเนื่อง 2 วินาทีภายใน 7 วินาที (ค่าเป้าหมายมาจากทฤษฎี 2 เท่าของเวลาปรับตัวเชิงทฤษฎี)")
    good = [r for r in runs if r["kind"] != "short"]
    table(doc, "balance_stats", "สถิติของการทรงตัว (เฉพาะรอบที่ตรวจจับลูกบอลได้ตลอด)",
          ["ตัวชี้วัด", "ค่าเฉลี่ย ± ส่วนเบี่ยงเบน", "ต่ำสุด–สูงสุด", "เป้าหมาย"],
          [["เวลาเข้าโซน 4 cm (s)", mean_sd([r["settle_s"] for r in good]), span([r["settle_s"] for r in good]), "≤ 7"],
           ["RMS position error (mm)", mean_sd([r["rms_mm"] for r in good]), span([r["rms_mm"] for r in good]), "≤ 10"],
           ["Steady-state error (mm)", mean_sd([r["steady_mm"] for r in good]), span([r["steady_mm"] for r in good]), "≤ 20"],
           ["Overshoot (%)", mean_sd([r["overshoot_pct"] for r in good], 0), span([r["overshoot_pct"] for r in good], 0), "≤ 20"],
           ["อัตราความสำเร็จ", rate_text(ok_count, len(runs)), "-", "≥ 80%"]],
          widths=[2.0, 1.8, 1.4, 1.0],
          note="ส่วนเบี่ยงเบนคำนวณจากตัวอย่าง (n − 1) ช่วงเชื่อมั่นเป็นแบบ Wilson เมื่อ n = 10 ช่วงกว้างมาก จึงควรอ่านเป็นแนวโน้ม ไม่ใช่ค่าที่แน่นอน")
    bins = [("ระยะเริ่ม ≤ 7 cm", lambda s: s <= 7.0), ("ระยะเริ่ม 7–9 cm", lambda s: 7.0 < s <= 9.0),
            ("ระยะเริ่ม > 9 cm (ไกลกว่าที่กล้องตรวจจับได้ดี)", lambda s: s > 9.0)]
    brow = []
    for label, cond in bins:
        sel = [(CF.start_info(r["stamp"])[0], r) for r in runs]
        sel = [r for s, r in sel if cond(s)]
        brow.append([label, str(len(sel)), rate_text(sum(1 for r in sel if r["success"]), len(sel)) if sel else "-"])
    table(doc, "balance_bins", "อัตราความสำเร็จแยกตามระยะที่วางลูกบอลตอนเริ่ม",
          ["กลุ่ม", "จำนวนรอบ", "สำเร็จ"], brow, widths=[3.0, 1.0, 2.2])
    figure(doc, figs["start_vs_result"], width=5.95)
    far_lost = sum(1 for r in runs if r["kind"] == "short" or r.get("lost"))
    max_start = max(CF.start_info(r["stamp"])[0] for r in runs)
    para(doc, f"ทรงตัวสำเร็จ {ok_count} จาก {len(runs)} รอบ วางลูกบอลอิสระ ระยะเริ่มไกลสุด {max_start:.1f} cm ({far_lost} รอบที่ลูกบอลหลุด) "
              "การทดลองนี้จึงไม่ได้ทดสอบระยะเกิน 9 cm ซึ่งเป็นขีดจำกัดที่กล้องตรวจจับลูกบอลได้ไม่สม่ำเสมอ (ขอบภาพสว่างจ้า) "
              "ผลของระยะเริ่มไกลกว่านี้จึงยังไม่มีข้อมูลสนับสนุน", indent=0.4)

    heading(doc, "4.3.3 ผลการเดินตามเส้นทาง", 2)
    para(doc, "สั่งให้เส้นอ้างอิงเริ่มจากศูนย์กลางแล้วค่อยๆ ขยายเป็นวงกลมหรือหกเหลี่ยมที่ครอบวงกลมรัศมี 3 cm คาบ 20 วินาทีต่อรอบ "
              "ความสำเร็จของการทดลองเส้นทางหมายถึงทำครบเวลาโดยไม่สูญเสียลูกบอล ส่วนความแม่นยำในการตามเส้นทางประเมินแยกด้วยความคลาดเฉลี่ยและรัศมีของวงที่ลูกบอลวน", indent=0.4)
    figure(doc, figs["real_paths"], width=5.95)
    rows = []
    path_stats = {}
    for key in ("circle", "hexagon"):
        rs = [r for r in scored(m, key) if r["kind"] == "path"]
        allr = scored(m, key)
        path_stats[key] = {"track": [r["track_mm"] for r in rs], "ratio": [r["radius_ratio"] for r in rs],
                           "angle": [r["angle_ok_pct"] for r in rs], "n": len(allr), "ok": sum(1 for r in allr if r["success"]),
                           "track_ok": sum(1 for r in rs if r["track_ok"])}
        ps = path_stats[key]
        rows.append([m[key]["label"], rate_text(ps["ok"], ps["n"]), mean_sd(ps["track"]), span(ps["track"]),
                     mean_sd(ps["ratio"], 2), f"{np.mean(ps['angle']):.0f}%" if ps["angle"] else "-",
                     f"{ps['track_ok']}/{len(rs)}"])
    table(doc, "paths", "ผลการเดินตามเส้นทางบนแพลตฟอร์มจริง",
          ["เส้นทาง", "สำเร็จ (ไม่สูญเสียลูกบอล)", "ความคลาดเทียบเวลาเดียวกัน (mm)", "ต่ำสุด–สูงสุด", "รัศมีวงบอล ÷ รัศมีที่สั่ง",
           "ทิศตรงกับเส้นทาง (±45°)", "ผ่านเป้า ≤ 15 mm"],
          rows, widths=[1.3, 1.5, 0.9, 0.8, 0.9, 0.8, 0.7],
          note="ความคลาดเฉลี่ยคือระยะเฉลี่ยระหว่างลูกบอลกับจุดที่ควรอยู่ ณ เวลาเดียวกัน ตัวควบคุมที่ปล่อยลูกบอลนิ่งที่ศูนย์กลางจะได้ความคลาดเท่ากับรัศมีเส้นทาง (30 mm) "
               "จึงตั้งเป้าให้ดีกว่านั้นอย่างน้อยครึ่งหนึ่ง (15 mm)")

    # ---- transient / steady split, mean +- SD, cross-track error
    import path_analysis as PA
    heading(doc, "4.3.4 ช่วงวิ่งเข้าหาเป้าหมาย ช่วงเกาะเส้นทาง และความคลาดเคลื่อนเชิงพื้นที่", 2)
    para(doc, "ลูกบอลถูกวางอิสระ ช่วงวินาทีแรกๆ จึงวิ่งเข้าหาศูนย์กลางจากจุดที่ต่างกัน การซ้อนทุกรอบเข้าด้วยกันตรงๆ จะดูไม่รู้เรื่อง "
              "จึงแยกวิเคราะห์เป็นสองช่วง ช่วงแรก (transient) คือ 6 วินาทีที่ทรงตัวที่ศูนย์กลางและ 5 วินาทีที่เส้นทางค่อยๆ ขยาย ใช้ดูความสามารถในการลู่เข้าหาเป้าหมาย (รูปที่ 4.9) "
              "ช่วงหลัง (steady state) คือช่วงเส้นทางเต็มขนาด ใช้เทียบกับทฤษฎี (รูปที่ 4.10)", indent=0.4)
    figure(doc, figs["transient"], width=5.95)
    conv = {}
    for key in ("circle", "hexagon"):
        grid, mu, sd, arr = PA.transient_curves(m[key]["runs"])
        conv[key] = float(np.nanmean(mu[(grid >= 3.0) & (grid < 6.0)]))
    para(doc, f"ในช่วง 3–6 วินาที (ทรงตัวที่ศูนย์กลาง) ลูกบอลอยู่ห่างศูนย์กลางเฉลี่ย {conv['circle']:.0f} mm (รอบวงกลม) และ {conv['hexagon']:.0f} mm (รอบหกเหลี่ยม) "
              "ซึ่งอยู่ในโซน 4 cm แต่ไม่ลงถึงศูนย์ตามทฤษฎี", indent=0.4)
    figure(doc, figs["polar_band"], width=5.95)
    para(doc, "การวัดความคลาดเคลื่อนแบบเทียบกับจุดอ้างอิง ณ เวลาเดียวกัน จะได้ค่าสูงเมื่อลูกบอลวิ่งตามหลังเป้าหมาย (phase lag) แม้จะวิ่งอยู่ใกล้เส้นทาง "
              "จึงวัดเพิ่มอีกแบบคือ ความคลาดเคลื่อนเชิงพื้นที่ (cross-track error) คือระยะสั้นที่สุดจากลูกบอลถึงเส้นทางอ้างอิง (วงกลม หรือขอบหกเหลี่ยม) "
              "คำนวณในช่วงเกาะเส้นทางของแต่ละรอบ แล้วนำ 10 รอบมาเฉลี่ย", indent=0.4)
    cross_rows, cross_stats = [], {}
    for key in ("circle", "hexagon"):
        lost_flags = _lost = [bool(r.get("lost")) for r in scored(m, key)]
        sums = [PA.run_summary(s_, key) for s_ in m[key]["runs"]]
        cx = [x_["cross_mm"] for x_ in sums]
        sy = [x_["sync_mm"] for x_ in sums]
        keep = [c for c, l_ in zip(cx, lost_flags) if not l_]
        cross_stats[key] = {"cross": cx, "sync": sy, "keep": keep, "lost": lost_flags,
                            "pass": sum(1 for c in cx if c <= RM._limit("track_mm"))}
        cross_rows.append([m[key]["label"], mean_sd(cx), span(cx), mean_sd(keep) if keep else "-", mean_sd(sy),
                           f"{cross_stats[key]['pass']}/{len(cx)}"])
    table(doc, "cross", "ความคลาดเคลื่อนเฉลี่ยของการเดินตามเส้นทาง (ช่วงเกาะเส้นทาง เฉลี่ยจาก 10 รอบ)",
          ["เส้นทาง", "Cross-track เฉลี่ย ± SD (mm)", "ต่ำสุด–สูงสุด", "เฉพาะรอบที่ลูกบอลไม่หลุด (mm)",
           "เทียบเวลาเดียวกัน (mm)", f"รอบที่ cross-track ≤ {RM._limit('track_mm'):g} mm"],
          cross_rows, widths=[1.5, 1.3, 0.9, 1.2, 1.0, 1.0],
          note=(f"เป้าหมาย {RM._limit('track_mm'):g} mm คือครึ่งหนึ่งของความคลาดของตัวควบคุมที่ปล่อยลูกบอลนิ่งที่ศูนย์กลาง (30 mm ทั้งแบบเทียบเวลาและแบบ cross-track "
                "เพราะเส้นทางอยู่ห่างศูนย์กลางเฉลี่ย 30 mm) ส่วนเกณฑ์ที่เข้มกว่า 10 mm ที่อาจใช้ในรายงานอื่น ไม่ผ่านทั้งสองเส้นทาง"))
    figure(doc, figs["cross_track"], width=5.95)

    heading(doc, "4.3.5 การเลือกค่าตัวอินทิเกรตสำหรับเส้นทาง (ทดลองก่อนแก้ตัวตรวจจับ)", 2)
    ladder = RM.ladder_summary(m["ladder_tags"], baseline=[RM.LOG_DIR / n for n in RM.LADDER_LEGACY_BASELINE]) if m["ladder_tags"] else None
    if ladder:
        base, chosen = ladder[0], ladder[-1]["chosen_ki"]
        para(doc, "ก่อนการทดลองหลักได้ทดลองค่า k_i หลายค่าบนวงกลม 3 cm ด้วยเกนชุดเดียวกัน (ค่าละ 2 รอบ) โดยกำหนดเกณฑ์เลือกไว้ล่วงหน้า: "
                  "เลือก k_i ที่มากที่สุดที่ (ก) ไม่ทำให้ลูกบอลหลุดเลย (ข) การเคลื่อนที่ย่าน 0.2–2 Hz ไม่เกิน 1.5 เท่าของกรณี k_i = 0 "
                  "และ (ค) รัศมีวงที่ลูกบอลวนและความคลาดเฉลี่ยดีกว่ากรณี k_i = 0 ทั้งสองอย่าง ถ้าไม่มีค่าใดผ่านให้ใช้ k_i = 0", indent=0.4)
        rows = [["0 (เทียบ)", str(base["n"]), "-", f"{base['radius']:.2f}", f"{base['track']:.1f}", f"{base['osc']:.2f}", "-"]]
        for r in ladder[1:-1]:
            rows.append([f"{r['ki']:g}", str(r["n"]), str(r["lost"]), f"{r['radius']:.2f}", f"{r['track']:.1f}", f"{r['osc']:.2f}",
                         "ผ่าน" if r["passes"] else "ไม่ผ่าน"])
        table(doc, "ladder", "ผลการทดลองเลือกค่า k_i (วงกลม 3 cm)",
              ["k_i", "รอบ", "ลูกบอลหลุด", "รัศมีวงบอล ÷ ที่สั่ง", "ความคลาด (mm)", "การเคลื่อนที่ 0.2–2 Hz (mm)", "ตามเกณฑ์"],
              rows, widths=[0.9, 0.5, 0.8, 1.1, 0.9, 1.3, 0.8],
              note=f"ค่าที่เลือกตามเกณฑ์: k_i = {chosen:g} จำนวน 2 รอบต่อค่า ตรวจได้เพียงความล้มเหลวรุนแรง (การแกว่ง ไม่เสถียร) แยกความต่างเล็กน้อยไม่ได้ "
                    "ข้อควรระวัง: การทดลองชุดนี้ทำก่อนแก้ปัญหาการตรวจจับในหัวข้อ 4.3.1 (ข้อมูลรอบที่ลูกบอลหายจากภาพมีความเร็วผิดพลาด) "
                    "และกรณีเทียบ k_i = 0 เป็นรอบเช้าที่เก็บก่อนหน้านั้นเช่นกัน จึงสรุปได้เพียงว่า ณ เวลานั้นตัวอินทิเกรตไม่ช่วย ยังไม่ได้ทดลองซ้ำหลังแก้")
    else:
        para(doc, f"การทดลองหลักใช้ k_i = {k_i:g} rad/(m·s)" + (" คือปิดตัวอินทิเกรต" if k_i == 0 else "") +
                  " (ยังไม่มีผลการทดลองเลือกค่า k_i ในชุดข้อมูลนี้)", indent=0.4)

    supp = [(key, scored(m, key)) for key in ("ellipse", "circle_5cm", "circle_7cm") if key in m]
    if supp:
        rows = []
        for key, rs in supp:
            rr = [r for r in rs if r["kind"] == "path" and r.get("path_seconds", 0) > 15]
            rows.append([m[key]["label"], str(len(rs)), str(sum(1 for r in rs if r.get("lost"))),
                         ", ".join(f"{r['radius_ratio']:.2f}" for r in rr) if rr else "-"])
        table(doc, "supp", "การทดลองเพิ่มเติมก่อนหน้า (เส้นทางและขนาดอื่น ใช้เกนชุดทรงตัว ยกเว้นวงรีที่ใช้เกนที่สูงกว่า)",
              ["เส้นทาง", "จำนวนรอบ", "ลูกบอลหลุด", "รัศมีวงบอล ÷ รัศมีที่สั่ง (รอบที่วิเคราะห์ได้)"], rows, widths=[2.0, 0.9, 1.0, 2.5],
              note="ผลเพิ่มเติมนี้เก็บก่อนแก้ปัญหาการตรวจจับในหัวข้อ 4.3.1 (ที่รัศมี 3 cm ลูกบอลวนได้เพียงราวหนึ่งในสามของรัศมีที่สั่ง เทียบกับ 0.6–0.8 หลังแก้) "
               "ส่วนที่ 5–7 cm วนถึงรัศมีใกล้ที่สั่งแต่หลุดบ่อย จึงตรึงขนาดเส้นทางที่ 3 cm ตัวเลขชุดนี้จึงใช้เป็นหลักฐานเปรียบเทียบก่อน-หลังแก้ได้ ไม่ใช่ผลหลัก")

    callout(doc, "สรุปผลบนแพลตฟอร์มจริง (4.3)", [
        f"ทรงตัวที่ศูนย์กลาง: สำเร็จ {rate_text(ok_count, len(runs))} ความคลาดขณะนิ่ง {span([r['steady_mm'] for r in good], 0)} mm",
        f"เดินตามวงกลม: ไม่สูญเสียลูกบอล {rate_text(path_stats['circle']['ok'], path_stats['circle']['n'])} Cross-track เฉลี่ย {mean_sd(cross_stats['circle']['cross'])} mm "
        f"(เทียบเวลาเดียวกัน {mean_sd(cross_stats['circle']['sync'])} mm) วนที่รัศมี {mean_sd(path_stats['circle']['ratio'], 2)} เท่าของที่สั่ง",
        f"เดินตามหกเหลี่ยม: ไม่สูญเสียลูกบอล {rate_text(path_stats['hexagon']['ok'], path_stats['hexagon']['n'])} Cross-track เฉลี่ย {mean_sd(cross_stats['hexagon']['cross'])} mm "
        f"(เทียบเวลาเดียวกัน {mean_sd(cross_stats['hexagon']['sync'])} mm) วนที่รัศมี {mean_sd(path_stats['hexagon']['ratio'], 2)} เท่าของที่สั่ง",
        f"เป้าหมายความคลาด {RM._limit('track_mm'):g} mm: ผ่านเมื่อวัดแบบ cross-track (เฉลี่ย) แต่ไม่ผ่านเมื่อวัดเทียบเวลาเดียวกัน และไม่ผ่านเกณฑ์ ≤ 10 mm",
    ])

    # ------------------------------------------------------- 4.4
    heading(doc, "4.4 ผลเปรียบเทียบระหว่างผลการจำลองและผลการทดลองจริง", 1)
    para(doc, "เปรียบเทียบผลจากแบบจำลองในหัวข้อ 4.2 กับผลจริงในหัวข้อ 4.3 ที่เกนชุดเดียวกัน ตารางที่ 4.15 สรุปตัวเลข "
              "ส่วนรูปที่ 4.12 และ 4.13 แสดงให้เห็นว่าแนวโน้มเหมือนหรือต่างกันตรงไหน", indent=0.4)
    sim_cross = {}
    for shape in ("circle", "hexagon"):
        Tm, Pm, _Rm = CF.sim_path(shape, g, k_i)
        sel_ = Tm > CF.HOLD_S + CF.RAMP_S
        sim_cross[shape] = float(PA.cross_track_m(Pm[sel_], PA.reference_polyline(shape)).mean() * 1000)
    rows = [
        ["Overshoot (%)", f"{sm1['overshoot_pct']:.0f}", span([r["overshoot_pct"] for r in good], 0), "ต่างกัน: ผลจริงเลยเป้ามากกว่า"],
        ["เวลาเข้าโซน 4 cm (s)", f"{sim_zone_settle(bal, 60, d_est):.1f}", span([r["settle_s"] for r in good]), "ใกล้เคียงกัน"],
        ["Steady-state error (mm)", "0", span([r["steady_mm"] for r in good], 0), "ต่างกัน: ผลจริงไม่ลงถึงศูนย์"],
        ["RMS error (mm)", "0", span([r["rms_mm"] for r in good], 0), "ต่างกัน"],
    ]
    for key in ("circle", "hexagon"):
        ps = path_stats[key]
        rows.append([f"ความคลาดเฉลี่ย {m[key]['label']} (mm)", f"{sim[key]['track']:.1f}", span(ps["track"]), "ต่างกันมาก"])
        rows.append([f"Cross-track เฉลี่ย {m[key]['label']} (mm)", f"{sim_cross[key]:.1f}", span(cross_stats[key]["cross"]),
                     "ต่างกัน แต่ผ่านเป้า 15 mm" if np.mean(cross_stats[key]["cross"]) <= RM._limit("track_mm") else "ต่างกัน ไม่ผ่านเป้า"])
        rows.append([f"รัศมีวงบอล ÷ ที่สั่ง ({m[key]['label']})", f"{sim[key]['ratio']:.2f}", span(ps["ratio"], 2), "ต่างกันมาก"])
        rows.append([f"ทิศ/จังหวะการวนตรงกับเส้นทาง ({m[key]['label']})", "ตรง",
                     (span(ps["angle"], 0) + " % ของเวลา") if ps["angle"] else "-", "เหมือนกัน"])
    table(doc, "compare", "เปรียบเทียบผลจำลองกับผลจริง",
          ["ตัวชี้วัด", "จำลอง (มีเวลาหน่วง)", "ผลจริง (ต่ำสุด–สูงสุดของทุกรอบ)", "แนวโน้ม"], rows,
          widths=[2.3, 1.1, 1.6, 1.4])
    figure(doc, figs["compare_balance"])
    figure(doc, figs["compare_radius"], width=5.95)
    heading(doc, "4.4.1 เหตุผลที่ผลจริงต่างจากผลจำลอง", 2)
    para(doc, "ส่วนที่เหมือนกันคือแนวโน้มของการเคลื่อนที่ ได้แก่ ลูกบอลเข้าหาศูนย์กลางด้วยเวลาใกล้เคียงกัน และวนตามทิศ "
              "และจังหวะของเส้นทาง ส่วนที่ต่างกันมี 4 เรื่อง", indent=0.4)
    heading(doc, "(1) แผ่นมีช่วงที่ไม่ตอบสนอง (dead zone) และแรงฝืด", 3)
    figure(doc, figs["open_loop"])
    para(doc, "แบบจำลองเชิงเส้นสมมติว่าเอียงแผ่นเท่าไรลูกบอลก็เคลื่อนตามสัดส่วนเสมอ แต่การทดสอบวงเปิดแสดงว่าที่มุมเล็กกว่าประมาณ 0.5° "
              "ลูกบอลแทบไม่เคลื่อน (แรงฝืดสถิตและระยะฟรีของข้อต่อ) การเดินตามเส้นทางรัศมี 3 cm คาบ 20 วินาทีต้องการมุมเอียงเพียง "
              f"{TH.reference_tilt_demand_deg(0.030, 20.0):.3f}° ซึ่งอยู่ลึกในช่วงนี้ ลูกบอลจึงเคลื่อนเป็นช่วงๆ และวนได้ไม่ตรงกับที่สั่ง "
              "ส่วนการเข้าสู่ศูนย์กลางในช่วงแรกใช้มุมใหญ่จึงเหมือนแบบจำลอง "
              "อย่างไรก็ดี หลังแก้ปัญหาการตรวจจับ (หัวข้อ 4.3.1) รัศมีวงที่ลูกบอลวนเพิ่มจากราวหนึ่งในสามเป็นประมาณ 0.6–0.9 เท่าของที่สั่ง "
              "แสดงว่าความต่างส่วนหนึ่งมาจากซอฟต์แวร์ ไม่ใช่กลไกทั้งหมด จึงไม่ควรอ้าง dead zone เป็นสาเหตุเดียว", indent=0.4)
    heading(doc, "(2) สัญญาณรบกวนของการวัดความเร็ว", 3)
    ns = command_noise_stats(m)
    para(doc, f"ความเร็วลูกบอลคำนวณจากผลต่างตำแหน่งระหว่างเฟรม สัญญาณรบกวนตำแหน่งจึงถูกขยายเป็นความเร็วคลาดประมาณ "
              f"{ns['vel_std']:.0f} mm/s ขณะที่ความเร็วจริงบนเส้นทางมีเพียงประมาณ 9–10 mm/s พลังงานของสัญญาณความเร็วกว่า "
              f"{ns['hi_frac']:.0f}% อยู่ในความถี่ที่ลูกบอลเคลื่อนตามไม่ได้ ทำให้คำสั่งมุมเอียงกลับทิศประมาณ {ns['rev']:.0f} ครั้งต่อวินาที "
              f"และพจน์ความเร็วในคำสั่ง ({ns['vel_term']:.2f}°) ใหญ่กว่าพจน์ตำแหน่ง ({ns['pos_term']:.2f}°) แบบจำลองเชิงเส้นไม่รวมผลนี้ "
              "อย่างไรก็ดี การสั่นนี้อาจช่วยให้แผ่นข้ามแรงฝืดได้ด้วย จึงยังสรุปไม่ได้ว่าการกรองสัญญาณจะทำให้ผลดีขึ้น", indent=0.4)
    heading(doc, "(3) เวลาหน่วงของลูป", 3)
    para(doc, "เวลาหน่วงทำให้ที่เผื่อเสถียรภาพลดลง (รูปที่ 4.2) การทดลองที่เพิ่มการหน่วงเฟสหรือเกนเทียบเท่าในลูป เช่น การเร่งตัวชดเชยความเอียง "
              "และการชดเชย dead zone ทำให้ระบบแกว่งจนไม่เสถียร (ตารางที่ 4.16) ซึ่งสอดคล้องกับที่เผื่อเหลือน้อย " +
              ("เวลาหน่วงที่วัดได้ยืนยันว่าที่เกนที่ใช้ปกติระบบยังเสถียร จึงไม่ใช่สาเหตุหลักของความต่างในผลปกติ" if is_measured else
               "แต่เวลาหน่วงจริงยังไม่ได้วัดตรง จึงยังระบุน้ำหนักของสาเหตุนี้ต่อความต่างของผลปกติไม่ได้"), indent=0.4)
    heading(doc, "(4) ความล่าช้า (phase lag) ของตัวควบคุมแบบป้อนกลับอย่างเดียว", 3)
    lags = {}
    for key in ("circle", "hexagon"):
        lags[key] = [r["phase_deg"] for r in scored(m, key) if r["kind"] == "path" and r["phase_deg"] == r["phase_deg"]]
    lag_deg = -float(np.mean(lags["circle"] + lags["hexagon"]))
    cs = cross_stats
    para(doc, "ตัวควบคุม LQR ในงานนี้เป็นแบบป้อนกลับเท่านั้น คือสั่งจากความคลาดที่เกิดขึ้นแล้ว ลูกบอลจึงตอบสนองหลังเป้าหมายเสมอ "
              f"จากการวัด ลูกบอลตามหลังเส้นทางเฉลี่ย {lag_deg:.0f}° ของหนึ่งรอบ (ประมาณ {lag_deg / 360 * PROF.PATH_PERIOD_S:.1f} วินาที) "
              f"ผลคือความคลาดแบบเทียบเวลาเดียวกัน ({mean_sd(cs['circle']['sync'] + cs['hexagon']['sync'])} mm) สูงกว่าความคลาดเชิงพื้นที่ "
              f"({mean_sd(cs['circle']['cross'] + cs['hexagon']['cross'])} mm) เกือบเท่าตัว ส่วนต่างนี้คือส่วนที่เกิดจากความล่าช้า "
              "ไม่ใช่การหลุดออกนอกเส้นทาง นอกจากนี้ลูกบอลวนรัศมีเฉลี่ยราว 0.8–0.9 เท่าของที่สั่ง (แต่ละรอบต่างกันมาก) และมุมเอียงที่เส้นทางต้องการเล็กมาก "
              "(ต่ำกว่า dead zone) จึงเป็นข้อจำกัดด้านการตอบสนองของแผ่นและเซอร์โวร่วมด้วย ผลทดลองใส่ตัวอินทิเกรต (หัวข้อ 4.3.5) "
              "ไม่ช่วยให้ดีขึ้น ส่วนการชดเชยล่วงหน้า (feedforward) จากการคำนวณต้องการมุมเอียงเพียง "
              f"{TH.reference_tilt_demand_deg(0.030, 20.0):.3f}° ซึ่งเล็กกว่า dead zone มาก จึงคาดว่าลำพังไม่ช่วยตราบใดที่ dead zone ยังอยู่", indent=0.4)
    table(doc, "tuning", "การทดลองปรับจูนที่ทำมาก่อนหน้า และผลที่ได้",
          ["วิธีที่ทดลอง", "ผล"],
          [["เพิ่มเกนตำแหน่งเป็น k = 0.15", "ความคลาดในการเดินตามเส้นทางไม่ดีขึ้น (ประมาณ 22 mm เท่าเดิม) และ PM ลดลง"],
           ["เร่งตัวชดเชยความเอียง (trim) ระหว่างเส้นทางที่เกนชุดเส้นทาง (ki 0.5)", "ลูกบอลหลุดออกจากเส้นทางภายใน 1–2 วินาที ทั้ง 2 รอบ (ไม่เสถียร)"],
           ["เพิ่มค่าคงที่ตามเครื่องหมายคำสั่งเพื่อชดเชย dead zone (0.5°, 1.0°)", "ลูกบอลแกว่งเกินเส้นทางที่ความถี่ประมาณ 1 Hz และหลุด (แย่ลง)"],
           ["ลดคาบส่งคำสั่งเซอร์โวจาก 0.10 เป็น 0.04 วินาที", "ทรงตัวจากจุดเริ่มไกลได้ (ดีขึ้น)"],
           ["ขยายรัศมีเส้นทางเป็น 5 และ 7 cm", "บอลวนถึงรัศมีที่สั่งแต่หลุดบ่อย ใช้ชดเชยไม่ได้อย่างเสถียร"]],
          widths=[3.1, 3.1])
    callout(doc, "สรุปการเปรียบเทียบ (4.4)", [
        "แนวโน้มเหมือนกัน: การเข้าสู่ศูนย์กลางในช่วงแรก และทิศ/จังหวะการวนตามเส้นทาง",
        "แนวโน้มต่างกัน: ผลจริงไม่นิ่งสนิทที่ศูนย์ และเดินตามเส้นทางได้ไม่ตรงเส้น (ตามหลัง รัศมีวนไม่คงที่และมักเล็กกว่าที่สั่ง) ขณะที่แบบจำลองเข้าใกล้เส้นทางเกือบสมบูรณ์",
        "สาเหตุที่มีหลักฐานสนับสนุนคือ dead zone/แรงฝืดของแผ่น และความล่าช้าของตัวควบคุมแบบป้อนกลับ รองลงมาคือสัญญาณรบกวนของการวัดความเร็วและเวลาหน่วง (น้ำหนักที่แน่ชัดยังไม่ได้ทดสอบแยก)",
    ], fill="FFF2CC")

    # =========================================================== 5
    heading(doc, "บทที่ 5 สรุปผลการดำเนินโครงงาน", 0)
    heading(doc, "5.1 สรุปผลการดำเนินโครงงาน", 1)
    bullets(doc, [
        f"ระบบควบคุมแบบ LQR ที่ออกแบบไว้เสถียรตามการวิเคราะห์ทางทฤษฎี (โพลอยู่ครึ่งซ้าย ผ่าน Routh-Hurwitz) "
        f"และเมื่อรวมเวลาหน่วง {d_est:.2f} วินาที ยังมี Phase margin {ST.margins(bal, d_est)['pm_deg']:.0f}°",
        f"ผลการจำลองด้วย Python: overshoot ประมาณ {sm1['overshoot_pct']:.0f}% เข้าโซนศูนย์กลาง 4 cm ใน {sim_zone_settle(bal, 60, d_est):.1f} วินาที "
        "และ steady-state error เป็นศูนย์",
        f"ผลบนแพลตฟอร์มจริง (ทรงตัว {len(runs)} ครั้ง): สำเร็จ {rate_text(ok_count, len(runs))} เข้าโซน 4 cm ใน {span([r['settle_s'] for r in good])} วินาที "
        f"ความคลาดขณะนิ่ง {span([r['steady_mm'] for r in good], 0)} mm",
        f"การเดินตามเส้นทาง: วงกลมไม่สูญเสียลูกบอล {rate_text(path_stats['circle']['ok'], path_stats['circle']['n'])} หกเหลี่ยม {rate_text(path_stats['hexagon']['ok'], path_stats['hexagon']['n'])} "
        f"ความคลาดเชิงพื้นที่เฉลี่ย {mean_sd(cross_stats['circle']['cross'])} mm (วงกลม) และ {mean_sd(cross_stats['hexagon']['cross'])} mm (หกเหลี่ยม) "
        f"แต่เมื่อเทียบกับจุดอ้างอิง ณ เวลาเดียวกันได้ {mean_sd(cross_stats['circle']['sync'])} และ {mean_sd(cross_stats['hexagon']['sync'])} mm "
        "ลูกบอลวนตามทิศและจังหวะของเส้นทางได้ แต่ตามหลังและรัศมีวนไม่คงที่ (เฉลี่ยราว 0.8–0.9 เท่าของที่สั่ง) ซึ่งต่างจากผลจำลองที่ตามเส้นทางได้เกือบสนิท ระบบจึงยังไม่ผ่านเกณฑ์การติดตามเส้นทางแบบเข้มงวด",
        "ความต่างระหว่างผลจำลองกับผลจริงมาจากความไม่เป็นเชิงเส้นของกลไก (dead zone และแรงฝืด) เป็นหลัก "
        "รองลงมาคือสัญญาณรบกวนของการวัดความเร็วและเวลาหน่วงของลูป ซึ่งแบบจำลองเชิงเส้นไม่ได้รวมไว้",
        "การเพิ่มเกน การเร่ง trim และการชดเชย dead zone แบบง่ายไม่ช่วยให้ตามเส้นทางดีขึ้น แต่การลดเวลาส่งคำสั่งเซอร์โวช่วยให้ทรงตัวจากจุดไกลได้ดีขึ้น",
    ])
    heading(doc, "5.2 ข้อเสนอแนะ", 1)
    heading(doc, "5.2.1 ข้อเสนอแนะในการนำผลไปใช้", 2)
    bullets(doc, [
        f"ใช้เกนชุดเดียว (k = {g[0]:g}, kv = {g[1]:g}) ทั้งการทรงตัวและการเดินตามเส้นทาง เพราะเสถียรและทำให้เทียบผลระหว่างการทดสอบได้",
        "วางลูกบอลให้ห่างจากศูนย์กลางไม่เกินประมาณ 7 cm และอย่าให้อยู่ที่ขอบบนของภาพ เพราะกล้องตรวจจับไม่ได้เป็นช่วงๆ",
        "ใช้เส้นทางขนาดเล็ก (รัศมี 3 cm) เพื่อความเสถียร และคาดหวังว่าลูกบอลจะตามหลังเส้นทางราว 1–2 วินาทีและรัศมีวนไม่คงที่",
        "ควรใช้แสงสม่ำเสมอบนแผ่น ลดจุดสว่างจ้าซึ่งทำให้การตรวจจับลูกบอลหลุด",
    ])
    heading(doc, "5.2.2 ข้อเสนอแนะสำหรับการศึกษาครั้งต่อไป", 2)
    bullets(doc, [
        ("ลดเวลาหน่วงของลูปที่วัดได้แล้ว (สายภาพและการสื่อสารเซอร์โว) เพื่อเพิ่มที่เผื่อเสถียรภาพ" if is_measured else
         "วัดเวลาหน่วงของลูปโดยตรง (ทดสอบการตอบสนองเชิงความถี่ด้วยสคริปต์ test_freq_response.py) เพื่อให้การวิเคราะห์เสถียรภาพไม่ต้องอาศัยการประมาณ"),
        "ลดระยะฟรีและแรงฝืดทางกลไก (ขันข้อต่อให้แน่น ตรวจความเรียบและระนาบของแผ่น) เพราะเป็นสาเหตุหลักของความต่างจากทฤษฎี",
        "ลดเวลาหน่วงของสายประมวลผลภาพ เพื่อให้เพิ่มเกนได้โดยไม่เสียเสถียรภาพ",
        "ตัวควบคุมแบบ LQI (เพิ่มตัวอินทิเกรต) ทดลองแล้วที่ k_i 0.05–0.20 ไม่ทำให้ตามเส้นทางดีขึ้นในฮาร์ดแวร์นี้ หากจะศึกษาต่อควรทำร่วมกับการลด dead zone "
        "และวางแผนเส้นทางให้มุมของหกเหลี่ยมโค้งมนขึ้นหรือช้าลงที่มุม เพื่อลดการกระชากของความคลาด (ตอนนี้ปัดมุมด้วยค่าเฉลี่ยเคลื่อนที่ 1/12 ของรอบแล้ว)",
        "ทดสอบตัวกรองสัญญาณความเร็ว (หรือตัวประมาณสถานะ) เพื่อลดสัญญาณรบกวน โดยตรวจสอบด้วยการทดลองจริงว่าไม่ทำให้แผ่นข้ามแรงฝืดได้ยากขึ้น",
        "เพิ่มจำนวนรอบทดสอบและกำหนดจุดวางลูกบอลให้ซ้ำได้ เพื่อให้ช่วงเชื่อมั่นของอัตราความสำเร็จแคบลง",
        "ศึกษาการชดเชย dead zone ที่ไม่เพิ่มเกนเทียบเท่าในลูป (เช่น การชดเชยแบบมีฮิสเทอรีซิส) เพราะวิธีชดเชยแบบง่ายที่ทดลองแล้วทำให้ระบบไม่เสถียร",
    ])

    # =========================================================== appendix
    heading(doc, "ภาคผนวก ไฟล์ข้อมูลที่ใช้ในบทที่ 4", 0)
    rows = []
    for key in ("balance", "circle", "hexagon"):
        rows.append([m[key]["label"], "จากป้ายกำกับ (tag) ในไฟล์ .json ข้างไฟล์ log" if m[key].get("from_tags") else "ข้อมูลชั่วคราวจาก report_runs.json",
                     ", ".join(s.replace("20260924T", "").replace("Z", "") for s in m[key]["runs"])])
    para(doc, "ไฟล์ log อยู่ในโฟลเดอร์ test_integrated_system/logs ชื่อ run_<เวลา>Z.csv พร้อมไฟล์ .json ที่บันทึกเงื่อนไขของแต่ละรอบ", size=SMALL, italic=True)
    for label, source, files in rows:
        para(doc, f"{label}: {source}", size=SMALL, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, after=1)
        para(doc, files, size=SMALLER, align=WD_ALIGN_PARAGRAPH.LEFT, after=6)
    para(doc, "สร้างเอกสารนี้ใหม่หลังทดลองเพิ่มด้วย python ch4_docx.py ตัวเลข ตาราง และกราฟทุกอย่างจะคำนวณใหม่จากไฟล์ log", size=SMALL, italic=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write chapters 4-5 of the project report as .docx")
    parser.add_argument("-o", "--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    print(f"wrote {build(args.out)}")


if __name__ == "__main__":
    main()
