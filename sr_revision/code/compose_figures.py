# -*- coding: utf-8 -*-
r"""Compose the two merged main figures from the regenerated time-split panels,
and refresh the supplementary trading-signal figures.

Layout of the originals (measured):
  Fig3_Group_Comparison_merged.png : 4820x1132 = 2400 + 20 + 2400 wide, 1050 + 82 tall
  Fig4_Step_Curves_merged.png      : 3020x 832 = 1500 + 20 + 1500 wide,  750 + 82 tall
so: horizontal hstack, 20 px gap, 82 px bottom strip carrying the (a)/(b) labels.
"""
import os
import shutil
from PIL import Image, ImageDraw, ImageFont

BASE = r"H:\老婆\v3"
PLOTS = os.path.join(BASE, "analysis", "ts_run", "Plots")
GAP = 20
STRIP = 82


def font(size=46):
    for p in [r"C:\ProgramData\anaconda3\Lib\site-packages\matplotlib\mpl-data\fonts\ttf\DejaVuSans.ttf",
              r"C:\Windows\Fonts\arial.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def flat(path):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        return bg
    return im.convert("RGB")


def merged(p1, p2, out):
    a, b = flat(p1), flat(p2)
    h = max(a.size[1], b.size[1])
    W = a.size[0] + GAP + b.size[0]
    canvas = Image.new("RGB", (W, h + STRIP), (255, 255, 255))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.size[0] + GAP, 0))
    d = ImageDraw.Draw(canvas)
    f = font(48)
    for label, x0, x1 in (("(a)", 0, a.size[0]), ("(b)", a.size[0] + GAP, W)):
        tw = d.textlength(label, font=f)
        d.text(((x0 + x1) / 2 - tw / 2, h + (STRIP - 48) / 2 - 6), label, fill=(0, 0, 0), font=f)
    canvas.save(out, dpi=(150, 150))
    print("wrote %-46s %dx%d" % (os.path.basename(out), canvas.size[0], canvas.size[1]))


def main():
    merged(os.path.join(PLOTS, "Group_Comparison_4assets_pv.png"),
           os.path.join(PLOTS, "Group_Comparison_6assets_pv.png"),
           os.path.join(BASE, "Fig3_Group_Comparison_merged.png"))

    merged(os.path.join(PLOTS, "learning_curves", "SARL_BERT_CM_AC_4assets_step_curve.png"),
           os.path.join(PLOTS, "learning_curves", "SARL_BERT_CM_AC_6assets_step_curve.png"),
           os.path.join(BASE, "Fig4_Step_Curves_merged.png"))

    # supplementary trading-signal figures
    src = os.path.join(PLOTS, "individual_stocks")
    dst = os.path.join(BASE, "supplementary")
    n = 0
    for f in os.listdir(src):
        if f.startswith("SARL_BERT") and f.endswith(".png"):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
            n += 1
    print("copied %d trading-signal figures to supplementary/" % n)

    # supplementary group-comparison panels (prefixed to avoid clashing with main)
    for k in ("2assets", "4assets", "6assets"):
        s = os.path.join(PLOTS, "Group_Comparison_%s_pv.png" % k)
        d = os.path.join(dst, "Group_Comparison_%s_pv.png" % k)
        shutil.copy2(s, d)
        print("copied Group_Comparison_%s_pv.png" % k)


if __name__ == "__main__":
    main()
