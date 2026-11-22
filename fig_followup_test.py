"""Fig. 13: predictions of both GP models fitted to the seven seed builds against the
measurements of the four follow-up builds, with 95% predictive intervals
(latent plus measurement variance).

Run from results/:  python ../code/fig_followup_test.py
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hybrid_gpr as g


def _builds_csv():
    """Locate builds.csv next to the script or in ../data (repository layout)."""
    here = Path(__file__).resolve().parent
    for p in (here / "builds.csv", here.parent / "data" / "builds.csv", Path.cwd() / "builds.csv"):
        if p.exists():
            return p
    raise FileNotFoundError("builds.csv not found next to the script or in ../data")


TARGETS = [("rho", "Density", "Density (g/cm$^3$)"),
           ("HV", "Hardness", "Hardness HV1"),
           ("logp", "Cross-sectional porosity", r"$\log_{10} p_{2D}$ (%)")]
MODELS = [("Zero-mean GP", False, "#7f7f7f", -0.09),
          ("Hybrid-GPR", True, "#1f4e79", 0.09)]
MEASURED = "#c55a11"


def main():
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    d = pd.read_csv(_builds_csv())
    seed, fu = d[d.campaign == "seed"], d[d.campaign != "seed"].reset_index(drop=True)
    Xf = g.features(fu.P, fu.v, fu.h)
    fits = {name: g.fit_all(seed, phys) for name, phys, _, _ in MODELS}
    x = np.arange(len(fu))
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.0))
    for ax, (t, title, ylab) in zip(axes, TARGETS):
        y, s = g.target_values(fu, t), g.noise(fu, t)
        for name, _, col, dx in MODELS:
            mu, sd = fits[name][t].predict(Xf)
            half = 1.96 * np.sqrt(sd ** 2 + s ** 2)
            ax.errorbar(x + dx, mu, yerr=half, fmt="o", ms=4, color=col, ecolor=col,
                        elinewidth=1.0, capsize=2.5, label=f"{name}, 95% predictive interval")
        ax.plot(x, y, "s", ms=6, color=MEASURED, label="Measured", zorder=5)
        ax.set_xticks(x, fu.build)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylab)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    handles, labels = handles[1:] + handles[:1], labels[1:] + labels[:1]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"figT_independent_test.{ext}", dpi=300 if ext == "png" else None)


if __name__ == "__main__":
    main()
