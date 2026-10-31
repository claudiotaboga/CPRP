"""CPRP house style — measured from CPRP-US8487456B2-20261031 (reference reel).
Canvas 1080x1920. Safe text column x120–900 (centre x=510).
Font: DejaVu Sans Bold / DejaVu Sans (as in the reference).
"""
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
CX = 510
SAFE_L, SAFE_R = 120, 900
B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
def f(path, size): return ImageFont.truetype(path, size)

WHITE, YELLOW, NAVY, TEAL, DARK = "#FFFFFF", "#F4C04E", "#0B1628", "#14BDB2", "#0B1628"
F_TITLE = f(B, 66)       # top hook/mechanism title, 2 lines, uppercase
F_SUB = f(B, 44)         # narration subtitles
F_FOOT = f(B, 30)        # CPRP / @curiouspatents
F_CONCEPT = f(R, 28)     # "Concept visualization"
F_PILL = f(B, 58)        # THE REAL PATENT
F_NUM = f(B, 58)         # patent number
F_DATE = f(B, 42)        # GRANTED dd MON yyyy (yellow)
F_ASSIGNEE = f(B, 36)    # assignee/inventor name
F_META = f(B, 23)        # ASSIGNEE • COUNTRY

Y_TITLE, Y_CONCEPT, Y_SUB, Y_FOOT = 290, 1265, 1355, 1500

def _center(d, y, txt, fnt, fill, stroke=0, sfill="#000000", spacing=6):
    bb = d.multiline_textbbox((0, 0), txt, font=fnt, spacing=spacing, align="center", stroke_width=stroke)
    w = bb[2] - bb[0]
    d.multiline_text((CX - w / 2 - bb[0], y - bb[1]), txt, font=fnt, fill=fill, spacing=spacing,
                     align="center", stroke_width=stroke, stroke_fill=sfill)
    return bb[3] - bb[1]

def wrap(d, txt, fnt, maxw):
    words, lines, cur = txt.split(), [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if d.textlength(t, font=fnt) <= maxw: cur = t
        else: lines.append(cur); cur = w_
    lines.append(cur); return "\n".join(lines)

def title(d, line1, line2=None, line2_yellow=False):
    """Uppercase white title with black outline; optional yellow 2nd line (used for hook questions)."""
    h = _center(d, Y_TITLE, line1.upper(), F_TITLE, WHITE, stroke=5)
    if line2:
        _center(d, Y_TITLE + h + 10, line2.upper(), F_TITLE, YELLOW if line2_yellow else WHITE, stroke=5)

def concept_label(d):
    _center(d, Y_CONCEPT, "Concept visualization", F_CONCEPT, WHITE, stroke=2)

def subtitle(d, txt):
    wt = wrap(d, txt, F_SUB, SAFE_R - SAFE_L - 30)
    bb = d.multiline_textbbox((0, 0), wt, font=F_SUB, spacing=6, align="center")
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    d.rectangle([CX - w / 2 - 14, Y_SUB - 12, CX + w / 2 + 14, Y_SUB + h + 16], fill=(17, 17, 17))
    d.multiline_text((CX - w / 2 - bb[0], Y_SUB - bb[1]), wt, font=F_SUB, fill=WHITE, spacing=6, align="center")

def footer(d):
    _center(d, Y_FOOT, "CPRP  /  @curiouspatents", F_FOOT, WHITE, stroke=3)

def reveal_card(img, number, date_label, name, meta, drawing):
    """Patent reveal: navy bg, teal pill, number, yellow date, name, meta, ORIGINAL drawing on white card.
    drawing: PIL image of the original patent drawing (or None → placeholder, never publishable)."""
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, H], fill=NAVY)
    d.rounded_rectangle([130, 270, 890, 408], 18, fill=TEAL)
    _center(d, 312, "THE REAL PATENT", F_PILL, NAVY)
    _center(d, 448, number, F_NUM, WHITE)
    _center(d, 522, date_label.upper(), F_DATE, YELLOW)
    _center(d, 566, name.upper(), F_ASSIGNEE, WHITE)
    _center(d, 610, meta.upper(), F_META, WHITE)
    d.rounded_rectangle([130, 636, 890, 1302], 22, fill=WHITE)
    if drawing is not None:
        dr = drawing.convert("RGB"); dr.thumbnail((720, 626))
        img.paste(dr, (CX - dr.width // 2, 969 - dr.height // 2))
    return d
