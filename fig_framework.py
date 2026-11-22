"""Fig. 1: one experimental feedback cycle. Vector drawing, 130 mm wide.
Run from results/: python ../code/fig_framework.py  ->  fig_framework.pdf, fig_framework.svg, fig_framework_preview.png"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

plt.rcParams.update({"font.family": "Liberation Sans", "pdf.fonttype": 42,
                     "svg.fonttype": "none", "mathtext.fontset": "custom",
                     "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
                     "font.size": 7.6})
NAVY, ORANGE, GREEN, INK = "#1F3A5F", "#B4581E", "#2E7D5B", "#1A1A1A"
BAND = "#C9DBEE"
BODY, TITLE = 7.6, 8.2

W_MM, H_MM = 130, 136
fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W_MM); ax.set_ylim(0, H_MM); ax.axis("off")


def box(x0, y0, x1, y1, col, dashed=False):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=1.6", fc="white", ec=col,
                                lw=0.9, ls=(0, (4, 2.5)) if dashed else "-"))


def title(x, y, n, text, col, size=TITLE):
    if n is not None:
        ax.add_patch(Circle((x + 2.3, y), 2.2, fc=col, ec="none"))
        ax.text(x + 2.3, y - 0.05, str(n), color="white", ha="center", va="center",
                fontsize=7.6, weight="bold")
        x += 5.6
    ax.text(x, y, text, color=col, ha="left", va="center", fontsize=size, weight="bold")


def t(x, y, s, **k):
    k.setdefault("ha", "center"); k.setdefault("va", "center"); k.setdefault("color", INK)
    k.setdefault("fontsize", BODY)
    ax.text(x, y, s, **k)


def arrow(p, q, col=NAVY, ls="-", rad=0.0, ms=7):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=col, lw=0.9,
                                 ls=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


P2D = r"p$_\mathregular{2D}$"
# columns
L0, L1 = 1.5, 35.5          # step 6
C0, C1 = 39.5, 92.5         # steps 1-5
R0, R1 = 96.0, 128.5        # targets, build 12
mid = (C0 + C1) / 2

# 1 Seed dataset
box(C0, 120, C1, 134, NAVY); title(C0 + 2, 130.6, 1, "Seed dataset", NAVY)
t(mid, 126.4, "S1–S7  |  N = 7", weight="bold")
t(mid, 122.7, "P, v, h  →  ρ, HV, " + P2D)
arrow((mid, 120), (mid, 116.5))

# 2 Hybrid-GPR
box(C0, 86, C1, 116.5, NAVY); title(C0 + 2, 113.0, 2, "Hybrid-GPR surrogates", NAVY)
xs = np.linspace(0, 1, 80)
px, py, cw = C0 + 5.5, 104.0, 16
ax.plot(px + cw * xs, py + 5.8 * np.exp(-((xs - 0.5) / 0.2) ** 2), color=NAVY, lw=1.0)
t(px + cw / 2, 101.4, "Prior mean m(x)")
arrow((px + cw + 2.2, 107.0), (px + cw + 7.0, 107.0), ms=6)
qx = px + cw + 9.2
m = py + 3.0 + 1.6 * np.sin(2 * np.pi * (xs * 1.1 + 0.1))
ax.fill_between(qx + cw * xs, m - 1.4, m + 1.4, color=BAND, lw=0)
ax.plot(qx + cw * xs, m, color=NAVY, lw=1.0)
t(qx + cw / 2, 101.4, "Posterior ± 1 SD")
t(mid, 96.9, "Prior mean + offset + residual GP")
t(mid, 93.1, "Noise: learned (initial model)", fontsize=7.3)
t(mid, 89.5, "→ fixed scales (present model)", fontsize=7.3)
arrow((mid, 86), (mid, 82.5))

# 3 Probabilistic decision
box(C0, 61.5, C1, 82.5, ORANGE); title(C0 + 2, 79.0, 3, "Probabilistic decision", ORANGE)
t(mid, 74.9, "Joint target probability", weight="bold")
t(mid, 71.3, "Cost-aware density CEI")
t(mid, 67.7, "Pareto / desirability")
t(mid, 64.4, "(post-processing)")
arrow((mid, 61.5), (mid, 58))

# Engineering targets
box(R0, 61.5, R1, 82.5, NAVY)
ax.text((R0 + R1) / 2, 79.0, "Engineering targets", color=NAVY, ha="center", va="center",
        fontsize=7.8, weight="bold")
t(R0 + 2.4, 74.4, "ρ ≥ 8.40 g/cm³", ha="left")
t(R0 + 2.4, 70.4, "HV ≥ 275", ha="left")
t(R0 + 2.4, 66.4, P2D + ": reported only", ha="left")
arrow((R0, 72.0), (C1 + 0.15, 72.0), col=ORANGE)

# 4 Four follow-up builds
box(C0, 40, C1, 58, ORANGE); title(C0 + 2, 54.6, 4, "Four follow-up builds", ORANGE)
for lab, txt, x, y in [("A", "Operating point", C0 + 3, 50.4), ("B", "Repeat of S2", C0 + 28.5, 50.4),
                       ("C", "CEI candidate", C0 + 3, 46.6), ("D", "Exploration", C0 + 28.5, 46.6)]:
    t(x, y, lab, ha="left", weight="bold", color=ORANGE)
    t(x + 3.3, y, txt, ha="left")
t(mid, 42.8, "The initial model guided A and C.", fontsize=7.2)
arrow((mid, 40), (mid, 36.5))

# 5 Manufacture and measure
box(C0, 18, C1, 36.5, NAVY); title(C0 + 2, 33.0, 5, "Manufacture and measure", NAVY)
t(mid, 28.6, "L-PBF on EOS M290")
t(mid, 25.0, "Density · HV1 · optical · CT")
t(mid, 21.3, "A: 8.425 g/cm³ · 283.9 HV1", fontsize=7.4)

# 6 Update & learn, with the hardness mini-plot
box(L0, 44, L1, 82.5, GREEN); title(L0 + 1.8, 79.0, 6, "Update & learn", GREEN)
t(L0 + 2.6, 74.6, "N = 7 → 11", ha="left", weight="bold")
hv = lambda v: 7.5 + (v - 265.0) / 25.0 * 25.5          # 265-290 HV -> x
thr = hv(275)
for yy in (66.4, 62.4):
    ax.plot([hv(265), hv(290)], [yy, yy], color="#9AA4AE", lw=0.6)
ax.plot([thr, thr], [60.6, 69.4], color=GREEN, lw=0.9, ls=(0, (2.5, 1.5)))
t(thr, 71.0, "275 HV1", fontsize=7.0, color=GREEN)
t(L0 + 2.6, 66.4, "A", ha="left", weight="bold", color=ORANGE)
t(L0 + 2.6, 62.4, "C", ha="left", weight="bold", color=ORANGE)
ax.plot(hv(283.9), 66.4, "o", ms=3.6, color=ORANGE, mec=ORANGE)
t(hv(283.9), 68.5, "283.9", fontsize=7.0)
ax.plot(hv(270.8), 62.4, "D", ms=3.2, color=ORANGE, mec=ORANGE)
t(hv(270.8), 64.5, "270.8", fontsize=7.0)
t(L0 + 2.6, 56.4, "Local hardness update", ha="left")
t(L0 + 2.6, 52.6, "Density level shift", ha="left")
t(L0 + 2.6, 48.8, "Top rank near A", ha="left")

# feedback arrows
arrow((C0, 27.2), ((L0 + L1) / 2, 44), col=GREEN, rad=-0.35)            # 5 -> 6
ax.plot([(L0 + L1) / 2] * 2, [82.5, 101.0], color=GREEN, lw=0.9)
arrow(((L0 + L1) / 2, 101.0), (C0, 101.0), col=GREEN)                     # 6 -> 2
t(L0 + 0.5, 104.0, "Measured feedback", ha="left", color=GREEN)

# Proposed build 12
box(R0, 23, R1, 56, NAVY, dashed=True); title(R0 + 2.2, 52.6, None, "Proposed build 12", NAVY, 7.8)
t(R0 + 2.4, 47.9, "With improved", ha="left")
t(R0 + 2.4, 44.5, "density protocol", ha="left")
t(R0 + 2.4, 39.6, "Top-ranked near A", ha="left", weight="bold")
t(R0 + 2.4, 35.4, "P = 262 W", ha="left")
t(R0 + 2.4, 31.6, "v = 1094 mm/s", ha="left")
t(R0 + 2.4, 27.8, "h = 0.071 mm", ha="left")

# next iteration (dashed): from step 6, along the bottom, up to build 12
bx = (R0 + R1) / 2
ax.plot([L0 + 5, L0 + 5, bx, bx], [44, 12.5, 12.5, 20.6], color=NAVY, lw=0.8, ls=(0, (3, 2)))
arrow((bx, 20.0), (bx, 23.0), col=NAVY, ms=6)
t(L0 + 7, 14.4, "next iteration", ha="left", color=NAVY, fontsize=7.2)

t(W_MM / 2, 5.5, "One experimental feedback cycle · 7 seed + 4 follow-up builds", fontsize=7.4)

for ext in ("pdf", "svg"):
    fig.savefig(f"fig_framework.{ext}")
fig.savefig("fig_framework_preview.png", dpi=220)
