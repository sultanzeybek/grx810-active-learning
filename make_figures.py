"""
Regenerate the analysis figures at 300 dpi into figures/.

Run:  python src/make_figures.py

Generates the three supplementary analysis figures (VED-porosity correlation,
cost-coefficient sensitivity, desirability surface). The main-text figures are
produced by the same pipeline; this script focuses on the analyses that are most
useful to reproduce independently.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from grx810_framework import (
    load_data, Hybrid, cost_aware_cei,
    prior_density, prior_hardness, prior_logp, LAYER_THICKNESS)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)
t = LAYER_THICKNESS

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.family": "serif",
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": "white", "figure.facecolor": "white"})
C = {"hgpr": "#1f4e79", "accent": "#2e7d32", "xgb": "#c55a11", "rf": "#7f7f7f"}


def fig_ved_p2d(df, X):
    VED, p2D = df.VED.values, df.p2D.values
    r1, _ = pearsonr(VED, p2D)
    vv = np.linspace(68, 110, 100)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.3))
    a1.scatter(VED, p2D, s=45, color=C["hgpr"], edgecolor="#111", linewidth=0.5, zorder=4)
    for i in range(len(df)):
        a1.annotate(f"S{df.S.values[i]}", (VED[i], p2D[i]), fontsize=6.5,
                    xytext=(4, 3), textcoords="offset points")
    a1b = a1.twinx()
    a1b.plot(vv, 10 ** np.clip(-0.8 + 8e-4 * (vv - 85) ** 2, -1.5, 0.5),
             color=C["xgb"], lw=1.4, ls="--")
    a1b.set_ylabel("prior shape (%)", color=C["xgb"], fontsize=8)
    a1.axvline(85, color="#999", lw=0.7, ls=":")
    a1.set_xlabel("VED (J/mm$^3$)"); a1.set_ylabel("Measured $p_{2D}$ (%)")
    a1.set_title(f"Porosity vs VED ($r={r1:+.2f}$)", fontsize=9)
    a2.scatter(VED, df.rho.values, s=45, color=C["accent"], edgecolor="#111", linewidth=0.5, zorder=4)
    for i in range(len(df)):
        a2.annotate(f"S{df.S.values[i]}", (VED[i], df.rho.values[i]), fontsize=6.5,
                    xytext=(4, 3), textcoords="offset points")
    a2.plot(vv, 8.20 + 0.24 * np.exp(-((vv - 85) ** 2) / (2 * 18.0 ** 2)),
            color=C["xgb"], lw=1.4, ls="--")
    a2.axvline(85, color="#999", lw=0.7, ls=":")
    a2.set_xlabel("VED (J/mm$^3$)"); a2.set_ylabel("Measured density (g/cm$^3$)")
    a2.set_title("Density vs VED (contrast)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "figS_ved_p2d.png"), bbox_inches="tight")
    print("  figS_ved_p2d.png")


def fig_cost(df, X):
    rng = np.random.default_rng(7); n = 10000
    P = rng.uniform(150, 400, n); v = rng.uniform(600, 1400, n); h = rng.uniform(0.06, 0.10, n)
    VED = P / (v * h * t); Cd = 1000 * np.sqrt(v) / P; TI = P / np.sqrt(v)
    G = np.column_stack([P, v, h, VED, Cd, TI])
    hr = Hybrid(prior_density).fit(X, df.rho.values)
    hh = Hybrid(prior_hardness).fit(X, df.HV.values)
    mu, sd = hr.predict(G, return_std=True); hv = hh.predict(G)
    feas = (mu >= 8.39) & (hv >= 275) & (VED < 100)
    coeffs = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    Ps, Vs = [], []
    for lam in coeffs:
        a = cost_aware_cei(mu, sd, df.rho.values.max(), feas, P, lam=lam)
        i = int(np.argmax(a)); Ps.append(P[i]); Vs.append(VED[i])
    pos = [(c, p, vv) for c, p, vv in zip(coeffs, Ps, Vs) if c > 0]
    cp, Pp, Vp = [x[0] for x in pos], [x[1] for x in pos], [x[2] for x in pos]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.2))
    a1.plot(cp, Pp, "o-", color=C["hgpr"], ms=5, label="$\\lambda>0$")
    a1.scatter([cp[0] / 3], [Ps[0]], marker="x", s=55, color=C["rf"], zorder=5, label="$\\lambda=0$")
    a1.set_xscale("log"); a1.axvline(0.005, color=C["xgb"], ls="--", lw=0.8); a1.set_ylim(215, 290)
    a1.set_xlabel("Cost coefficient $\\lambda$"); a1.set_ylabel("Recommended $P^*$ (W)")
    a1.set_title("Recommended power vs $\\lambda$", fontsize=9); a1.legend(fontsize=7, loc="lower right")
    a2.plot(cp, Vp, "s-", color=C["accent"], ms=5)
    a2.scatter([cp[0] / 3], [Vs[0]], marker="x", s=55, color=C["rf"], zorder=5)
    a2.set_xscale("log"); a2.axvline(0.005, color=C["xgb"], ls="--", lw=0.8); a2.set_ylim(83, 86)
    a2.set_xlabel("Cost coefficient $\\lambda$"); a2.set_ylabel("VED$^*$ (J/mm$^3$)")
    a2.set_title("Energy density vs $\\lambda$", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "figS_cost_sensitivity.png"), bbox_inches="tight")
    print("  figS_cost_sensitivity.png")


def fig_desirability(df, X):
    def d_LTB(y, lo, hi): return np.clip((y - lo) / (hi - lo), 0, 1)
    def d_STB(y, lo, hi): return np.clip((hi - y) / (hi - lo), 0, 1)
    hr = Hybrid(prior_density).fit(X, df.rho.values)
    hh = Hybrid(prior_hardness).fit(X, df.HV.values)
    hp = Hybrid(prior_logp).fit(X, df.logp.values)
    Pg = np.linspace(150, 400, 200); Vg = np.linspace(600, 1400, 200)
    PP, VV = np.meshgrid(Pg, Vg)
    Pf, vf = PP.ravel(), VV.ravel(); hf = np.full_like(Pf, 0.08)
    VEDf = Pf / (vf * hf * t); Cdf = 1000 * np.sqrt(vf) / Pf; TIf = Pf / np.sqrt(vf)
    G = np.column_stack([Pf, vf, hf, VEDf, Cdf, TIf])
    rg = hr.predict(G); hg = hh.predict(G); pg = 10 ** hp.predict(G)
    Dg = (d_LTB(rg, 8.20, 8.44) * d_STB(pg, 0.05, 0.90) * d_LTB(hg, 270, 285)) ** (1 / 3)
    Dg = Dg.reshape(PP.shape)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    c = ax.contourf(VV, PP, Dg, levels=20, cmap="viridis")
    fig.colorbar(c, ax=ax, label="Desirability $D$")
    ax.contour(VV, PP, VEDf.reshape(PP.shape), levels=[85], colors="white",
               linewidths=0.9, linestyles="--")
    ax.scatter(df.v, df.P, s=28, facecolor="white", edgecolor="#222",
               linewidth=0.7, zorder=4, label="Training")
    ax.scatter([1100], [265], s=110, marker="*", color=C["xgb"], edgecolor="#111",
               linewidth=0.6, zorder=6, label="Recommended")
    ax.set_xlabel("Scan speed $v$ (mm/s)"); ax.set_ylabel("Laser power $P$ (W)")
    ax.set_title("Desirability surface ($h=0.08$ mm)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "figS_desirability.png"), bbox_inches="tight")
    print("  figS_desirability.png")


if __name__ == "__main__":
    df, X = load_data(os.path.join(HERE, "data", "builds.csv"))
    print("Generating figures at 300 dpi:")
    fig_ved_p2d(df, X)
    fig_cost(df, X)
    fig_desirability(df, X)
    print("Done. See figures/.")
