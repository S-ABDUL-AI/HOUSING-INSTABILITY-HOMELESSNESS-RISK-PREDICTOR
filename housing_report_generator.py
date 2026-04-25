"""
housing_report_generator.py
McKinsey-style PDF report for Housing Stability Risk Monitor
Uses two-PDF merge strategy to prevent dark cover bleeding onto body pages.
"""

import io
from datetime import date
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable, Image)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.pdfgen import canvas as rl_canvas

try:
    from pypdf import PdfWriter, PdfReader
except ImportError:
    from PyPDF2 import PdfWriter, PdfReader

# ── colour tokens ──────────────────────────────────────────────
NAVY     = colors.HexColor("#0A1F44")
NAVY_MID = colors.HexColor("#152B5C")
GOLD     = colors.HexColor("#C9A84C")
GOLD_LT  = colors.HexColor("#E8C97A")
OFF_WHITE= colors.HexColor("#F8F6F1")
INK      = colors.HexColor("#1A1A1A")
BODY_C   = colors.HexColor("#2C3E50")
MUTED    = colors.HexColor("#6B7280")
RED      = colors.HexColor("#C8382A")
AMBER    = colors.HexColor("#B8560A")
GREEN    = colors.HexColor("#1A7A2E")
RULE     = colors.HexColor("#E2E6EC")
WHITE    = colors.white

PW, PH = letter   # 612 × 792 pt
MARGIN  = 0.65 * inch
CW      = PW - 2 * MARGIN

# ── helper: matplotlib chart → reportlab Image ─────────────────
def fig_to_rl_image(fig, w_in, h_in):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=w_in*inch, height=h_in*inch)

# ── cover page ─────────────────────────────────────────────────
def _build_cover(buf):
    c = rl_canvas.Canvas(buf, pagesize=letter)
    # dark background
    c.setFillColor(NAVY)
    c.rect(0, 0, PW, PH, fill=1, stroke=0)
    # gold left stripe
    c.setFillColor(GOLD)
    c.rect(0, 0, 6, PH, fill=1, stroke=0)
    # geometric accent top-right
    c.setFillColor(NAVY_MID)
    c.rect(PW-160, PH-160, 160, 160, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#1E3A6E"))
    c.rect(PW-100, PH-100, 100, 100, fill=1, stroke=0)
    # gold rule under accent
    c.setFillColor(GOLD)
    c.rect(MARGIN, PH-180, CW, 2, fill=1, stroke=0)
    # eyebrow
    c.setFillColor(GOLD)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(MARGIN, PH-210, "HOUSING STABILITY RISK MONITOR  ·  GOVERNMENT PROGRAM OFFICERS")
    # title
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 28)
    c.drawString(MARGIN, PH-265, "Which communities are at greatest")
    c.drawString(MARGIN, PH-300, "risk of housing instability?")
    # subtitle
    c.setFillColor(GOLD_LT)
    c.setFont("Helvetica", 13)
    c.drawString(MARGIN, PH-336, "Evidence-based risk rankings and resource allocation")
    c.drawString(MARGIN, PH-354, "for HUD grantees and homelessness prevention programs")
    # gold rule
    c.setFillColor(GOLD)
    c.rect(MARGIN, PH-380, CW, 1.5, fill=1, stroke=0)
    # meta block
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(GOLD)
    c.drawString(MARGIN, PH-406, "PREPARED BY")
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(MARGIN, PH-424, "Sherriff Abdul-Hamid")
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.HexColor("#B0BFD8"))
    c.drawString(MARGIN, PH-442, "Product Leader  ·  Government Digital Services  ·  Safety Net Benefits Delivery")
    c.drawString(MARGIN, PH-458, "USAID  ·  UNDP  ·  UKAID  ·  Obama Foundation Leaders Award (Top 1.3%)")
    # date
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(GOLD)
    c.drawString(MARGIN, PH-490, "REPORT DATE")
    c.setFont("Helvetica", 10)
    c.setFillColor(WHITE)
    c.drawString(MARGIN, PH-506, date.today().strftime("%B %d, %Y"))
    # footer rule
    c.setFillColor(GOLD)
    c.rect(MARGIN, 52, CW, 1, fill=1, stroke=0)
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#6B7A9A"))
    c.drawString(MARGIN, 38, "HOUSING STABILITY RISK MONITOR  ·  CONFIDENTIAL — FOR PROGRAMME OFFICERS ONLY")
    c.showPage()
    c.save()

# ── white-background canvas class ─────────────────────────────
class _WhiteBgCanvas(rl_canvas.Canvas):
    def showPage(self):
        self.setFillColor(OFF_WHITE)
        self.rect(0, 0, PW, PH, fill=1, stroke=0)
        super().showPage()

class _WhiteBgMaker:
    def __call__(self, *args, **kwargs):
        return _WhiteBgCanvas(*args, **kwargs)

# ── page frame ─────────────────────────────────────────────────
def _draw_frame(c, doc):
    c.setFillColor(GOLD)
    c.rect(0, PH-4, PW, 4, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.rect(0, PH-32, PW, 28, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(GOLD)
    c.drawString(MARGIN, PH-22, "HOUSING STABILITY RISK MONITOR")
    c.setFillColor(WHITE)
    c.setFont("Helvetica", 8)
    pg = f"Page {doc.page}"
    c.drawRightString(PW-MARGIN, PH-22, pg)
    c.setFillColor(RULE)
    c.rect(0, 24, PW, 1, fill=1, stroke=0)
    c.setFont("Helvetica", 7)
    c.setFillColor(MUTED)
    src = "Data: HUD Fair Market Rents + Census ACS 5-year estimates + Princeton Eviction Lab"
    c.drawString(MARGIN, 12, src)
    c.drawRightString(PW-MARGIN, 12, date.today().strftime("%B %d, %Y"))

def _get_styles():
    ss = getSampleStyleSheet()
    def s(name, **kw):
        return ParagraphStyle(name, parent=ss["Normal"], **kw)
    return {
        "eyebrow": s("eyebrow", fontSize=8, textColor=GOLD, fontName="Helvetica-Bold",
                     spaceAfter=4, leading=12),
        "h1":      s("h1", fontSize=18, textColor=NAVY, fontName="Helvetica-Bold",
                     spaceAfter=6, leading=22),
        "h2":      s("h2", fontSize=13, textColor=NAVY, fontName="Helvetica-Bold",
                     spaceAfter=4, leading=17),
        "body":    s("body", fontSize=10, textColor=BODY_C, spaceAfter=6, leading=14),
        "small":   s("small", fontSize=8, textColor=MUTED, spaceAfter=4, leading=11),
        "caption": s("caption", fontSize=8, textColor=MUTED, fontName="Helvetica-Oblique",
                     spaceAfter=8, leading=11),
    }

# ── chart helpers ──────────────────────────────────────────────
def _band_color(band):
    return {"High":"#C8382A","Medium":"#B8560A","Low":"#1A7A2E"}.get(band,"#6B7280")

def _risk_bar_chart(df):
    top15 = df.nlargest(15, "risk_score")
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor="white")
    colors_list = [_band_color(b) for b in top15["risk_label"]]
    bars = ax.barh(top15["region"], top15["risk_score"], color=colors_list, height=0.6)
    for bar, val in zip(bars, top15["risk_score"]):
        ax.text(val+0.5, bar.get_y()+bar.get_height()/2,
                f"{val:.0f}", va="center", ha="left", fontsize=8, color="#1A1A1A")
    ax.set_xlabel("Risk Score (0–100)", fontsize=9, color="#2C3E50")
    ax.set_xlim(0, 115)
    ax.invert_yaxis()
    ax.tick_params(axis="both", labelsize=8, colors="#2C3E50")
    ax.spines[:].set_visible(False)
    ax.set_facecolor("white")
    legend_patches = [mpatches.Patch(color=_band_color(b), label=b)
                      for b in ["High","Medium","Low"]]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, frameon=False)
    plt.tight_layout()
    return fig

def _band_dist_chart(df):
    counts = df["risk_label"].value_counts().reindex(["High","Medium","Low"]).fillna(0)
    fig, ax = plt.subplots(figsize=(3.5, 3.0), facecolor="white")
    ax.bar(counts.index, counts.values,
           color=[_band_color(b) for b in counts.index], width=0.5)
    for x, v in zip(range(len(counts)), counts.values):
        ax.text(x, v+1, str(int(v)), ha="center", fontsize=9, color="#1A1A1A", fontweight="bold")
    ax.set_ylabel("Regions", fontsize=9)
    ax.spines[:].set_visible(False)
    ax.set_facecolor("white")
    ax.tick_params(axis="both", labelsize=9, colors="#2C3E50")
    plt.tight_layout()
    return fig

def _scatter_chart(df):
    fig, ax = plt.subplots(figsize=(7.5, 3.5), facecolor="white")
    for band in ["High","Medium","Low"]:
        sub = df[df["risk_label"]==band]
        ax.scatter(sub["cost_burden"], sub["eviction_rate"],
                   c=_band_color(band), label=band, alpha=0.7, s=25, edgecolors="none")
    ax.set_xlabel("Cost Burden Ratio", fontsize=9, color="#2C3E50")
    ax.set_ylabel("Eviction Rate (%)", fontsize=9, color="#2C3E50")
    ax.spines[:].set_visible(False)
    ax.set_facecolor("white")
    ax.tick_params(axis="both", labelsize=8, colors="#2C3E50")
    ax.legend(fontsize=8, frameon=False, title="Risk Band", title_fontsize=8)
    plt.tight_layout()
    return fig

# ── KPI table helper ───────────────────────────────────────────
def _kpi_table(kpis, styles):
    cells = [[Paragraph(f'<font color="#C9A84C"><b>{lab}</b></font><br/>'
                        f'<font size=18 color="#0A1F44"><b>{val}</b></font><br/>'
                        f'<font size=8 color="#6B7280">{sub}</font>', styles["small"])
              for lab, val, sub in kpis]]
    t = Table([cells], colWidths=[CW/len(kpis)]*len(kpis))
    t.setStyle(TableStyle([
        ("BOX",      (0,0),(-1,-1), 0.5, RULE),
        ("INNERGRID",(0,0),(-1,-1), 0.5, RULE),
        ("BACKGROUND",(0,0),(-1,-1), WHITE),
        ("VALIGN",   (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1), 10),
        ("BOTTOMPADDING",(0,0),(-1,-1),10),
        ("LEFTPADDING",(0,0),(-1,-1), 12),
    ]))
    return t

# ── main build function ────────────────────────────────────────
def build_housing_report_bytes(df: pd.DataFrame,
                                model_accuracy: float = 0.79,
                                budget_m: float = 25.0) -> bytes:
    # ── cover ──────────────────────────────────────────────────
    cover_buf = io.BytesIO()
    _build_cover(cover_buf)
    cover_buf.seek(0)

    # ── body ───────────────────────────────────────────────────
    body_buf = io.BytesIO()
    doc = SimpleDocTemplate(body_buf, pagesize=letter,
                            leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=0.55*inch, bottomMargin=0.55*inch,
                            canvasmaker=_WhiteBgMaker())
    styles = _get_styles()
    story  = []

    def add(el): story.append(el)
    def br(n=1):
        for _ in range(n): story.append(Spacer(1, 8))
    def rule():
        story.append(HRFlowable(width=CW, thickness=1, color=RULE, spaceAfter=10))

    # ── Page 1: Executive Snapshot ─────────────────────────────
    add(Paragraph("EXECUTIVE SNAPSHOT", styles["eyebrow"]))
    add(Paragraph("Housing Stability Risk Monitor — Panel Summary", styles["h1"]))
    rule()

    band_counts = df["risk_label"].value_counts()
    high_n  = int(band_counts.get("High",   0))
    med_n   = int(band_counts.get("Medium", 0))
    low_n   = int(band_counts.get("Low",    0))
    pct_h   = round(high_n / len(df) * 100, 1)
    med_rent= int(df["median_rent"].median())
    med_unemp= round(df["unemployment"].median(), 1)

    kpis = [
        ("REGIONS ANALYSED",  f"{len(df):,}", "Full panel"),
        ("HIGH-RISK",         str(high_n),    f"{pct_h}% of panel"),
        ("MEDIAN RENT",       f"${med_rent:,}","HUD FMR estimate"),
        ("UNEMPLOYMENT",      f"{med_unemp}%", "Census ACS 5-yr"),
        ("MODEL ACCURACY",    f"{int(model_accuracy*100)}%","Hold-out accuracy"),
    ]
    add(_kpi_table(kpis, styles))
    br()

    # band distribution
    fig_d = _band_dist_chart(df)
    add(Paragraph("Risk Band Distribution", styles["h2"]))
    add(fig_to_rl_image(fig_d, 3.8, 3.0))
    br()
    add(Paragraph(
        f"Of the {len(df):,} regions analysed, <b>{high_n}</b> ({pct_h}%) are "
        f"classified as <font color='#C8382A'><b>High</b></font> risk, "
        f"<b>{med_n}</b> as Medium, and <b>{low_n}</b> as Low. "
        "High-risk classification is driven by cost burdens above 40%, "
        "eviction rates above 5%, and unemployment above 7%.",
        styles["body"]))
    rule()

    # Band definitions table
    add(Paragraph("Priority Band Definitions", styles["h2"]))
    band_def = [
        ["Band", "Score Range", "Driver Criteria", "Recommended Action"],
        ["High",   "67–100", "Cost burden > 40%; eviction > 5%; unemployment > 7%",
         "Activate emergency rehousing & rental assistance"],
        ["Medium", "34–66",  "1–2 elevated indicators; affordability pressure emerging",
         "Accelerate SNAP outreach & eviction prevention funding"],
        ["Low",    "0–33",   "All indicators within moderate range",
         "Monitor affordability trends; maintain preventive access"],
    ]
    t_band = Table(band_def, colWidths=[0.7*inch, 0.9*inch, 2.8*inch, 2.3*inch])
    t_band.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,0),  NAVY),
        ("TEXTCOLOR",    (0,0),(-1,0),  GOLD),
        ("FONTNAME",     (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0,0),(-1,-1), 8),
        ("BACKGROUND",   (0,1),(0,1),   colors.HexColor("#FFF5F5")),
        ("BACKGROUND",   (0,2),(0,2),   colors.HexColor("#FFFBF0")),
        ("BACKGROUND",   (0,3),(0,3),   colors.HexColor("#F0FFF4")),
        ("TEXTCOLOR",    (0,1),(0,1),   RED),
        ("TEXTCOLOR",    (0,2),(0,2),   AMBER),
        ("TEXTCOLOR",    (0,3),(0,3),   GREEN),
        ("FONTNAME",     (0,1),(-1,-1), "Helvetica"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, colors.HexColor("#F8F9FC")]),
        ("BOX",          (0,0),(-1,-1), 0.5, RULE),
        ("INNERGRID",    (0,0),(-1,-1), 0.5, RULE),
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",   (0,0),(-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1), 6),
        ("LEFTPADDING",  (0,0),(-1,-1), 8),
    ]))
    add(t_band)

    # ── Page 2: Policy Brief + Budget Simulation ───────────────
    add(Spacer(1, 18))
    add(Paragraph("POLICY BRIEF", styles["eyebrow"]))
    add(Paragraph("Decision Summary — Risk · Implication · Action", styles["h1"]))
    rule()

    top3 = df.nlargest(3,"risk_score")["region"].tolist()
    sim_per = round(budget_m * 1e6 / max(high_n, 1) / 1000, 0)

    brief_data = [
        [Paragraph('<font color="#C8382A"><b>⚠ RISK</b></font>', styles["h2"]),
         Paragraph('<font color="#0A1F44"><b>→ IMPLICATION</b></font>', styles["h2"]),
         Paragraph('<font color="#1A7A2E"><b>✓ ACTION NOW</b></font>', styles["h2"])],
        [Paragraph(
            f"<b>{high_n} regions ({pct_h}%)</b> are classified High risk — driven by "
            f"cost burdens above 40%, eviction rates above 5%, and unemployment above 7%. "
            f"Top at-risk regions: {', '.join(top3)}.", styles["body"]),
         Paragraph(
            f"At the current ${budget_m:.0f}M simulation budget, each High-risk region "
            f"receives approximately <b>${sim_per:,.0f}K</b> — potentially insufficient "
            f"for regions with severe cost burden. Medium-risk regions ({med_n} total) "
            "require preventive investment to avoid escalation.", styles["body"]),
         Paragraph(
            "Activate rapid rehousing and rental assistance in High-risk regions. "
            "Accelerate SNAP outreach and eviction prevention in Medium-risk areas. "
            "Link disbursements to measurable housing stability milestones. "
            "Train the model to generate region-level ML predictions.", styles["body"])],
    ]
    t_brief = Table(brief_data, colWidths=[CW/3]*3)
    t_brief.setStyle(TableStyle([
        ("BOX",         (0,0),(-1,-1), 0.5, RULE),
        ("INNERGRID",   (0,0),(-1,-1), 0.5, RULE),
        ("BACKGROUND",  (0,0),(0,-1),  colors.HexColor("#FFF5F5")),
        ("BACKGROUND",  (1,0),(1,-1),  colors.HexColor("#F0F4FF")),
        ("BACKGROUND",  (2,0),(2,-1),  colors.HexColor("#F0FFF4")),
        ("VALIGN",      (0,0),(-1,-1), "TOP"),
        ("TOPPADDING",  (0,0),(-1,-1), 10),
        ("BOTTOMPADDING",(0,0),(-1,-1),10),
        ("LEFTPADDING", (0,0),(-1,-1), 10),
    ]))
    add(t_brief)
    br(2)

    # budget simulation
    add(Paragraph("BUDGET SIMULATION", styles["eyebrow"]))
    add(Paragraph(f"How does ${budget_m:.0f}M flow across risk bands?", styles["h2"]))
    weights = {"High":0.60,"Medium":0.30,"Low":0.10}
    sim_rows = [["Risk Band","Allocation ($M)","Regions","Per Region ($K)"]]
    for band, w in weights.items():
        alloc  = round(budget_m * w, 2)
        n_reg  = int(band_counts.get(band, 1))
        per_r  = round(alloc * 1e6 / max(n_reg,1) / 1000, 0)
        sim_rows.append([band, f"${alloc:.2f}M", str(n_reg), f"${int(per_r):,}K"])
    t_sim = Table(sim_rows, colWidths=[1.2*inch, 1.5*inch, 1.2*inch, 1.5*inch])
    t_sim.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,0), NAVY),
        ("TEXTCOLOR",    (0,0),(-1,0), GOLD),
        ("FONTNAME",     (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTNAME",     (0,1),(-1,-1),"Helvetica"),
        ("FONTSIZE",     (0,0),(-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, colors.HexColor("#F8F9FC")]),
        ("BOX",          (0,0),(-1,-1), 0.5, RULE),
        ("INNERGRID",    (0,0),(-1,-1), 0.5, RULE),
        ("TOPPADDING",   (0,0),(-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1), 6),
        ("LEFTPADDING",  (0,0),(-1,-1), 8),
    ]))
    add(t_sim)

    # ── Page 3: Regional Rankings ──────────────────────────────
    add(Spacer(1, 18))
    add(Paragraph("EXHIBIT 1 — REGIONAL RISK RANKINGS", styles["eyebrow"]))
    add(Paragraph("Top 15 highest-risk regions by composite score", styles["h1"]))
    rule()
    fig_r = _risk_bar_chart(df)
    add(fig_to_rl_image(fig_r, 7.5, 4.2))
    add(Paragraph(
        "Risk scores are composite indices incorporating cost burden (30%), "
        "income level (25%), unemployment (20%), eviction rate (15%), and "
        "household crowding (10%). Scores range from 0 (lowest risk) to 100 (highest risk).",
        styles["caption"]))
    br()

    # ── Page 4: Indicator Scatter + Rankings Table ─────────────
    add(Paragraph("EXHIBIT 2 — COST BURDEN vs EVICTION RATE", styles["eyebrow"]))
    add(Paragraph("Risk band distribution across key indicators", styles["h2"]))
    fig_s = _scatter_chart(df)
    add(fig_to_rl_image(fig_s, 7.5, 3.5))
    add(Paragraph(
        "Regions in the upper-right quadrant (high cost burden + high eviction rate) "
        "are most vulnerable to housing instability cascades. Targeted intervention "
        "in these areas yields the highest stabilisation impact per dollar spent.",
        styles["caption"]))
    br()

    # full rankings table (top 20)
    add(Paragraph("Regional Rankings — Top 20", styles["h2"]))
    top20 = df.nlargest(20,"risk_score")[["region","risk_label","risk_score",
                                          "median_rent","median_income",
                                          "unemployment","eviction_rate"]].copy()
    top20.insert(0,"Rank",range(1,len(top20)+1))
    top20["median_rent"]   = top20["median_rent"].apply(lambda x: f"${int(x):,}")
    top20["median_income"] = top20["median_income"].apply(lambda x: f"${int(x):,}")
    top20["unemployment"]  = top20["unemployment"].apply(lambda x: f"{x:.1f}%")
    top20["eviction_rate"] = top20["eviction_rate"].apply(lambda x: f"{x:.1f}%")
    top20["risk_score"]    = top20["risk_score"].apply(lambda x: f"{x:.0f}")
    tbl_data = [["Rank","Region","Band","Score","Rent","Income","Unemp.","Eviction"]]
    for _, r in top20.iterrows():
        tbl_data.append([str(r["Rank"]), r["region"], r["risk_label"],
                         r["risk_score"], r["median_rent"], r["median_income"],
                         r["unemployment"], r["eviction_rate"]])
    t_top = Table(tbl_data, colWidths=[0.4*inch,1.5*inch,0.6*inch,0.5*inch,
                                        0.7*inch,0.9*inch,0.6*inch,0.7*inch])
    band_color_map = {"High": colors.HexColor("#FFF5F5"),
                      "Medium": colors.HexColor("#FFFBF0"),
                      "Low":    colors.HexColor("#F0FFF4")}
    style_cmds = [
        ("BACKGROUND",   (0,0),(-1,0), NAVY),
        ("TEXTCOLOR",    (0,0),(-1,0), GOLD),
        ("FONTNAME",     (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTNAME",     (0,1),(-1,-1),"Helvetica"),
        ("FONTSIZE",     (0,0),(-1,-1), 7.5),
        ("BOX",          (0,0),(-1,-1), 0.5, RULE),
        ("INNERGRID",    (0,0),(-1,-1), 0.5, RULE),
        ("TOPPADDING",   (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING",  (0,0),(-1,-1), 5),
    ]
    for i, row in enumerate(tbl_data[1:], 1):
        bg = band_color_map.get(row[2], WHITE)
        style_cmds.append(("BACKGROUND",(0,i),(-1,i),bg))
    t_top.setStyle(TableStyle(style_cmds))
    add(t_top)

    # ── Page 5: Regional Insight Cards ────────────────────────
    add(Spacer(1, 18))
    add(Paragraph("REGIONAL INSIGHT CARDS", styles["eyebrow"]))
    add(Paragraph("Top 6 highest-risk regions — plain-language briefings", styles["h1"]))
    rule()

    def recommended_action(band):
        return {
            "High":   "Activate emergency housing stabilisation — prioritise rapid rehousing and rental assistance.",
            "Medium": "Accelerate SNAP outreach, affordable housing pipeline, and eviction prevention funding.",
            "Low":    "Monitor housing affordability trends and maintain preventive service access.",
        }.get(band, "")

    def why_region(row):
        flags = []
        if row.cost_burden   > 0.40: flags.append("severe cost burden (rent > 40% of income)")
        if row.eviction_rate > 5.0:  flags.append(f"elevated eviction rate ({row.eviction_rate:.1f}%)")
        if row.unemployment  > 7.0:  flags.append(f"high unemployment ({row.unemployment:.1f}%)")
        if row.crowding_rate > 0.10: flags.append(f"household crowding ({row.crowding_rate*100:.1f}%)")
        if not flags:
            return "Indicators within moderate range; monitor for emerging pressure."
        return "Elevated risk driven by: " + "; ".join(flags) + "."

    top6 = df.nlargest(6,"risk_score")
    for _, row in top6.iterrows():
        bc = {"High": colors.HexColor("#FFF5F5"),
              "Medium": colors.HexColor("#FFFBF0"),
              "Low": colors.HexColor("#F0FFF4")}.get(row.risk_label, WHITE)
        lc = {"High": RED, "Medium": AMBER, "Low": GREEN}.get(row.risk_label, MUTED)
        card_data = [[
            Paragraph(f'<b>{row.region}</b> &nbsp; '
                      f'<font color="{lc.hexval() if hasattr(lc,"hexval") else "#C8382A"}">'
                      f'Risk Score: {row.risk_score:.0f} — {row.risk_label}</font>',
                      styles["h2"]),
        ],[
            Paragraph(
                f'<font color="#6B7280">Rent: ${int(row.median_rent):,} &nbsp;·&nbsp; '
                f'Income: ${int(row.median_income):,} &nbsp;·&nbsp; '
                f'Unemployment: {row.unemployment:.1f}% &nbsp;·&nbsp; '
                f'Eviction: {row.eviction_rate:.1f}%</font><br/><br/>'
                f'<b>Why:</b> {why_region(row)}<br/>'
                f'<b>Recommended action:</b> {recommended_action(row.risk_label)}',
                styles["body"]),
        ]]
        t_card = Table(card_data, colWidths=[CW])
        t_card.setStyle(TableStyle([
            ("BACKGROUND",   (0,0),(-1,-1), bc),
            ("BOX",          (0,0),(-1,-1), 0.5, lc),
            ("LEFTPADDING",  (0,0),(-1,-1), 12),
            ("RIGHTPADDING", (0,0),(-1,-1), 12),
            ("TOPPADDING",   (0,0),(-1,-1), 8),
            ("BOTTOMPADDING",(0,0),(-1,-1), 8),
        ]))
        add(t_card)
        add(Spacer(1, 8))

    # ── Page 6: Methodology & Scope ───────────────────────────
    add(Spacer(1, 18))
    add(Paragraph("METHODOLOGY & SCOPE NOTE", styles["eyebrow"]))
    add(Paragraph("Data sources, model design, and limitations", styles["h1"]))
    rule()

    meth_rows = [
        ["Indicator", "Source", "Weight in Risk Score"],
        ["Median rent / Fair Market Rent", "HUD FMR (annual county estimates)", "25%"],
        ["Median household income",        "Census ACS 5-year",                  "30%"],
        ["Unemployment rate",              "Census ACS 5-year",                  "20%"],
        ["Eviction rate (proxy)",          "Princeton Eviction Lab",              "15%"],
        ["Household crowding",             "Census ACS 5-year",                  "10%"],
    ]
    t_meth = Table(meth_rows, colWidths=[2.2*inch, 2.8*inch, 1.7*inch])
    t_meth.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,0), NAVY),
        ("TEXTCOLOR",    (0,0),(-1,0), GOLD),
        ("FONTNAME",     (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTNAME",     (0,1),(-1,-1),"Helvetica"),
        ("FONTSIZE",     (0,0),(-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, colors.HexColor("#F8F9FC")]),
        ("BOX",          (0,0),(-1,-1), 0.5, RULE),
        ("INNERGRID",    (0,0),(-1,-1), 0.5, RULE),
        ("TOPPADDING",   (0,0),(-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1), 6),
        ("LEFTPADDING",  (0,0),(-1,-1), 8),
    ]))
    add(t_meth)
    br()

    limitations = [
        ["Limitation", "Impact", "Mitigation"],
        ["Synthetic/demo data", "Scores do not reflect real regional conditions",
         "Connect live HUD + Census APIs for production deployment"],
        ["Annual data lag", "FMR and ACS data may be 1–2 years behind current conditions",
         "Supplement with real-time eviction and unemployment feeds"],
        ["County-level aggregation", "Within-county variation is masked",
         "Disaggregate to tract level where ACS 5-yr data is available"],
        ["Equal indicator weighting", "Weights are illustrative; real deployments need calibration",
         "Calibrate weights against local eviction and homelessness outcomes data"],
    ]
    add(Paragraph("Model Limitations & Mitigations", styles["h2"]))
    t_lim = Table(limitations, colWidths=[1.5*inch, 2.5*inch, 2.7*inch])
    t_lim.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,0), NAVY),
        ("TEXTCOLOR",    (0,0),(-1,0), GOLD),
        ("FONTNAME",     (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTNAME",     (0,1),(-1,-1),"Helvetica"),
        ("FONTSIZE",     (0,0),(-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, colors.HexColor("#F8F9FC")]),
        ("BOX",          (0,0),(-1,-1), 0.5, RULE),
        ("INNERGRID",    (0,0),(-1,-1), 0.5, RULE),
        ("TOPPADDING",   (0,0),(-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1), 6),
        ("LEFTPADDING",  (0,0),(-1,-1), 8),
        ("VALIGN",       (0,0),(-1,-1), "TOP"),
    ]))
    add(t_lim)
    br()
    add(Paragraph(
        "<b>Scope note:</b> All data in this report is illustrative. Any real housing "
        "programme allocation decision must be paired with official HUD data, Census "
        "Bureau statistics, legal review, and agency approval processes. This tool is "
        "designed to support — not replace — government decision-making.",
        styles["body"]))
    br()
    add(HRFlowable(width=CW, thickness=1, color=GOLD, spaceAfter=8))
    add(Paragraph(
        f"<b>Prepared by Sherriff Abdul-Hamid</b> — Product leader specializing in "
        f"government digital services, safety net benefits delivery, and decision-support "
        f"tools for underserved communities. Former Founder & CEO, Poverty 360 "
        f"(25,000+ beneficiaries, West Africa). Directed $200M+ in resource allocation for "
        f"USAID, UNDP, and UKAID. <b>Obama Foundation Leaders Award</b> (Top 1.3%) · "
        f"<b>Mandela Washington Fellow</b> (Top 0.3%) · Harvard Business School. "
        f"Report generated: {date.today().strftime('%B %d, %Y')}.",
        styles["small"]))

    doc.build(story, onFirstPage=_draw_frame, onLaterPages=_draw_frame)
    body_buf.seek(0)

    # ── merge cover + body ─────────────────────────────────────
    writer = PdfWriter()
    for r in [PdfReader(cover_buf), PdfReader(body_buf)]:
        for page in r.pages:
            writer.add_page(page)
    out = io.BytesIO()
    writer.write(out)
    out.seek(0)
    return out.read()
