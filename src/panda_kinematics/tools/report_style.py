#!/usr/bin/env python3
"""
Shared helpers for the two Challenge 2 PDFs (make_report.py = study guide, make_submission_report.py =
from the recorded runs in challenge2_results/) and all figures.
"""
import gzip
import os
import re
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

TMP = tempfile.mkdtemp()

# ---------------------------------------------------------------- fonts / styles
FD = "/usr/share/fonts/truetype/dejavu/"
for name, f in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"),
                ("DV-I", "DejaVuSans-Oblique.ttf"), ("DV-BI", "DejaVuSans-BoldOblique.ttf"),
                ("DVM", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(name, FD + f))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

NAVY, TEAL, GREY = colors.HexColor("#1f3a5f"), colors.HexColor("#1b7f79"), colors.HexColor("#555555")
body = ParagraphStyle("body", fontName="DV", fontSize=9.6, leading=14, spaceAfter=5)
small = ParagraphStyle("small", parent=body, fontSize=8.4, leading=11.5, textColor=GREY)
h1 = ParagraphStyle("h1", fontName="DV-B", fontSize=16, leading=20, textColor=NAVY, spaceBefore=6, spaceAfter=8)
h2 = ParagraphStyle("h2", fontName="DV-B", fontSize=11.5, leading=15, textColor=TEAL, spaceBefore=8, spaceAfter=4)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=3, spaceAfter=2)
code = ParagraphStyle("code", fontName="DVM", fontSize=8.4, leading=11, backColor=colors.HexColor("#f2f4f7"),
                      borderPadding=5, spaceBefore=3, spaceAfter=7, leftIndent=4)
cell = ParagraphStyle("cell", parent=body, fontSize=8.6, leading=11.2, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="DV-B")
caption = ParagraphStyle("caption", parent=small, alignment=TA_CENTER, spaceAfter=8)

story = []


def H1(t):
    story.append(Paragraph(t, h1))


def H2(t):
    story.append(Paragraph(t, h2))


def P(t):
    story.append(Paragraph(t, body))


def B(items):
    for it in items:
        story.append(Paragraph(it, bullet, bulletText="•"))
    story.append(Spacer(1, 3))


def C(t):
    story.append(Preformatted(t, code))


def box(title, text, bg="#e8f4f3", edge="#1b7f79"):
    inner = [Paragraph(f"<b>{title}</b>", cell), Spacer(1, 2)] + \
            [Paragraph(x, cell) for x in (text if isinstance(text, list) else [text])]
    t = Table([[inner]], colWidths=[16.6 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
                           ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(edge)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.append(t)
    story.append(Spacer(1, 7))


def table(rows, widths, header=True, size=8.6):
    data = [[Paragraph(str(c), cellb if (header and i == 0) else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c0cc")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe7f1")))
    t.setStyle(TableStyle(st))
    story.append(t)
    story.append(Spacer(1, 7))


def fig(path, width_cm, cap):
    from PIL import Image as PI
    w, h = PI.open(path).size
    img = Image(path, width=width_cm * cm, height=width_cm * cm * h / w)
    story.append(KeepTogether([img, Paragraph(cap, caption)]))


def f3(v):
    return "[" + ", ".join(f"{x:+.3f}" for x in v) + "]"
