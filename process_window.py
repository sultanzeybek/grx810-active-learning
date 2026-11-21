"""Reproduce the seven-build feasible-region comparison, Tables 8 and 9 and Figures 9, 10 and 12.

Run from results/: python ../code/process_window.py --with-trees (for every row of Table 8).
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.stats import norm

import hybrid_gpr as g


CENTRELINE = np.array([275., 1000., .080])
DESIGN_STEPS = np.array([50., 200., .010])
PLANE_SIZE = 401


def inputs(frame):
    return g.features(frame.P, frame.v, frame.h)


def predictions(models, points):
    return {target: model.predict(points) for target, model in models.items()}


def feasible_means(field, density=g.RHO_THR, hardness=g.HV_THR, porosity=None):
    region = (field["rho"][0] >= density) & (field["HV"][0] >= hardness)
    if porosity is not None:
        region &= field["logp"][0] < np.log10(porosity)
    return region


def probability(field, density=g.RHO_THR, hardness=g.HV_THR, porosity=None, future=False):
    """Independent Gaussian targets; a future hardness measurement averages five indents."""
    mu_r, sd_r = field["rho"]
    mu_h, sd_h = field["HV"]
    if future:
        sd_r = np.hypot(sd_r, g.S_RHO)
        sd_h = np.hypot(sd_h, g.S_HV1 / np.sqrt(5))
    result = norm.cdf((mu_r - density) / sd_r) * norm.cdf((mu_h - hardness) / sd_h)
    if porosity is not None:
        mu_p, sd_p = field["logp"]
        if future:
            sd_p = np.hypot(sd_p, g.S_P)
        result *= norm.cdf((np.log10(porosity) - mu_p) / sd_p)
    return result


def fixed_prior(points):
    return {target: (fn(points), 0.) for target, fn in g.PRIORS.items()}


def tree_reference(seed, points):
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor

    fits = {"Random forest": RandomForestRegressor(n_estimators=50, max_depth=3, random_state=0, n_jobs=1),
            "XGBoost": XGBRegressor(n_estimators=50, max_depth=3, learning_rate=.1, random_state=0, n_jobs=1)}
    for name, model in fits.items():
        field = {}
        for target in ("rho", "HV"):
            model.fit(inputs(seed), g.target_values(seed, target))
            field[target] = (model.predict(points), None)
        yield name, field


def evaluate_box(data, seed, models, with_trees):
    p, v, h = g.candidates()
    points = g.features(p, v, h)
    hull = g.in_hull(seed, p, v, h)
    fields = {label: predictions(model, points) for label, model in models.items()}
    fields["Prior only"] = fixed_prior(points)
    if with_trees:
        fields.update(tree_reference(seed, points))
    comparison, masks = [], {}
    for name, field in fields.items():
        masks[name] = region = feasible_means(field)
        for domain, subset in (("full_box", np.ones(len(hull), dtype=bool)), ("seed_hull", hull)):
            chosen = region & subset
            centroid = points[chosen, :4].mean(axis=0) if chosen.any() else [np.nan] * 4
            comparison.append({"model": name, "criterion": "means", "domain": domain,
                               "n_domain": int(subset.sum()), "n_pass": int(chosen.sum()),
                               "percent": 100 * chosen.sum() / subset.sum(),
                               **dict(zip(("mean_power", "mean_speed", "mean_hatch", "mean_VED"), centroid))})
            if name not in models and name != "Prior only":
                continue
            kinds = ("future",) if name == "Prior only" else ("latent", "future")
            for kind in kinds:
                probs = probability(field, future=(kind == "future"))[subset]
                comparison.append({"model": name, "criterion": kind, "domain": domain,
                                   "n_domain": len(probs), "n_pass": int((probs >= .5).sum()),
                                   "percent": 100 * (probs >= .5).mean(), "peak": probs.max()})
    pd.DataFrame(comparison).to_csv("window_coverage.csv", index=False)
    for name, domain in (("full box", np.ones(len(hull), dtype=bool)), ("seed hull", hull)):
        both = (masks["Hybrid-GPR"] & masks["Prior only"] & domain).sum()
        union = ((masks["Hybrid-GPR"] | masks["Prior only"]) & domain).sum()
        print(f"{name}: intersection {both}, union {union}, IoU {both/union:.5f}")
    hybrid = fields["Hybrid-GPR"]
    centroid = points[masks["Hybrid-GPR"], :3].mean(axis=0)
    print(f"Full-box Hybrid-GPR centroid in design steps from centreline: "
          f"{np.linalg.norm((centroid - CENTRELINE)/DESIGN_STEPS):.5f}")
    centre = g.features(*[np.array([a]) for a in CENTRELINE])
    for label, model in models.items():
        f = predictions(model, centre)
        print(f"{label}, exact centreline: latent {probability(f)[0]:.6f}, "
              f"future measurement {probability(f, future=True)[0]:.6f}")
    at_a = predictions(models["Hybrid-GPR"], inputs(data.loc[data.build == "A"]))
    settings = [(r, h, None) for r in (8.35, 8.40, 8.45) for h in (270, 275, 280)]
    settings += [(g.RHO_THR, g.HV_THR, p) for p in (.15, .20, .25)]
    sensitivity = []
    for r, hv, por in settings:
        mask = feasible_means(hybrid, r, hv, por)
        sensitivity.append({"density_threshold": r, "hardness_threshold": hv, "porosity_limit_percent": por,
                            "box_count": int(mask.sum()), "box_percent": 100 * mask.mean(),
                            "hull_count": int((mask & hull).sum()), "hull_size": int(hull.sum()),
                            "max_latent_probability": probability(hybrid, r, hv, por).max(),
                            "latent_probability_at_A": probability(at_a, r, hv, por)[0]})
    pd.DataFrame(sensitivity).to_csv("window_thresholds.csv", index=False)


def slice_grid(hatch, size):
    power_edges = np.linspace(150, 400, size + 1)
    speed_edges = np.linspace(600, 1400, size + 1)
    power = (power_edges[1:] + power_edges[:-1]) / 2
    speed = (speed_edges[1:] + speed_edges[:-1]) / 2
    v, p = np.meshgrid(speed, power)
    return speed, power, g.features(p.ravel(), v.ravel(), np.full(p.size, hatch))


def axes_style(ax):
    ax.set(xlim=(600, 1400), ylim=(150, 400), xticks=(600, 800, 1000, 1200, 1400),
           yticks=(150, 200, 250, 300, 350, 400))
    ax.tick_params(labelsize=7.5, direction="out", length=3)
    ax.set_box_aspect(.84)


def save(fig, stem):
    for fmt in ("pdf", "svg", "png"):
        fig.savefig(stem + "." + fmt, dpi=360, bbox_inches="tight", pad_inches=.065)
    plt.close(fig)


def slice_figures(data, seed, models):
    speed, power, grid = slice_grid(.080, PLANE_SIZE)
    middle = {name: predictions(model, grid) for name, model in models.items()}
    middle["Prior only"] = fixed_prior(grid)
    plane_seeds = seed.loc[np.isclose(seed.h, .080)]
    xy = plane_seeds[["v", "P"]].to_numpy()
    vertices = ConvexHull(xy).vertices
    polygon = xy[np.r_[vertices, vertices[0]]]
    rows = []
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(9.15, 3.7))
    fig.subplots_adjust(left=.07, right=.984, bottom=.255, top=.84, wspace=.12)
    for ax, label, letter in zip(axes, ("Zero-mean GP", "Hybrid-GPR", "Prior only"), "abc"):
        mask = feasible_means(middle[label]).reshape(PLANE_SIZE, PLANE_SIZE)
        count = int(mask.sum())
        rows.append({"model": label, "hatch": .080, "cells_per_axis": PLANE_SIZE,
                     "passing_cells": count, "all_cells": mask.size, "percent": mask.mean()*100})
        ax.fill(polygon[:,0], polygon[:,1], color="#dcebd2", lw=0, zorder=0)
        if count:
            ax.contourf(speed, power, mask.astype(float), levels=(.5, 1.5), colors=("#83b5d4",), zorder=1)
        ax.plot(polygon[:,0], polygon[:,1], color="#4b5861", ls="--", lw=1.2, zorder=2)
        ax.plot(speed, 85 * speed * .080 * g.T_LAYER, color="#a94f1f", lw=1.2, ls=":")
        ax.scatter(xy[:,0], xy[:,1], s=15, facecolors="white", edgecolors="#1b2932", lw=.85, zorder=4)
        axes_style(ax)
        ax.set_title(f"({letter})  {label}\n{mask.mean()*100:.1f}% of this slice", fontsize=9, pad=6)
    axes[0].set_ylabel("Laser power (W)")
    fig.supxlabel("Scan speed (mm/s)", y=.144, fontsize=9)
    fig.legend(handles=[Patch(facecolor="#83b5d4", label="Both mean thresholds met"),
                        Patch(facecolor="#dcebd2", edgecolor="#4b5861", ls="--", label="Seed hull"),
                        Line2D([], [], color="#a94f1f", ls=":", label="VED = 85 J/mm³"),
                        Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec="#1b2932", label="Seed at this hatch")],
               loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(.52,-.005))
    save(fig, "figW_process_window")
    pd.DataFrame(rows).to_csv("window_slice.csv", index=False)

    _, _, grid_a = slice_grid(.071, PLANE_SIZE)
    panels = [(predictions(models["Hybrid-GPR"], grid_a), .071, "Hybrid-GPR"),
              (middle["Hybrid-GPR"], .080, "Hybrid-GPR"), (middle["Zero-mean GP"], .080, "Zero-mean GP")]
    fig = plt.figure(figsize=(9.5, 3.8))
    gs = fig.add_gridspec(1, 4, width_ratios=(1, 1, 1, .045), left=.07, right=.95,
                          bottom=.255, top=.84, wspace=.16)
    axes = [fig.add_subplot(gs[0,i]) for i in range(3)]
    levels = np.arange(0, .6001, .025)
    for i, (ax, (field, h, label)) in enumerate(zip(axes, panels)):
        probs = probability(field).reshape(PLANE_SIZE, PLANE_SIZE)
        contour = ax.contourf(speed, power, probs, levels=levels, cmap="Blues")
        mask = feasible_means(field).reshape(PLANE_SIZE, PLANE_SIZE)
        if mask.any() and not mask.all():
            ax.contour(speed, power, mask.astype(float), levels=(.5,), colors=("#bf5518",), linewidths=1)
        line_y = 85.9375 * speed * h * g.T_LAYER
        ax.plot(speed, line_y, ":", color="white", lw=1.85)
        ax.plot(speed, line_y, ":", color="#655c56", lw=.95)
        seeds_here = seed.loc[np.isclose(seed.h, h)]
        ax.scatter(seeds_here.v, seeds_here.P, facecolors="white", edgecolors="#152732", s=13, lw=.75, zorder=4)
        measured = data.loc[(data.campaign=="followup") & np.isclose(data.h, h)]
        for row in measured.itertuples():
            ax.scatter(row.v, row.P, marker="D", facecolors="#ffda8d", edgecolors="#392915", s=23, lw=.8, zorder=5)
            label_text = "S2/B" if row.build == "B" else row.build
            ax.annotate(label_text, (row.v,row.P), xytext=(5,6), textcoords="offset points", fontsize=7,
                        bbox={"facecolor":"white","alpha":.80,"edgecolor":"none","pad":.5}, zorder=6)
        axes_style(ax)
        ax.tick_params(labelleft=i==0)
        ax.set_title(f"({chr(97+i)})  {label}\nh = {h:.3f} mm", fontsize=9, pad=6)
    cb = fig.colorbar(contour, cax=fig.add_subplot(gs[0,3]), ticks=np.arange(0,.61,.1))
    cb.ax.tick_params(labelsize=7.5)
    cb.set_label("Joint latent probability", fontsize=8)
    axes[0].set_ylabel("Laser power (W)")
    fig.supxlabel("Scan speed (mm/s)", y=.144, fontsize=9)
    fig.legend(handles=[Line2D([],[],color="#bf5518",lw=1.2,label="Mean-based region boundary"),
                        Line2D([],[],color="#655c56",ls=":",label="VED = 85.9 J/mm³"),
                        Line2D([],[],ls="",marker="o",ms=4,mec="#152732",mfc="white",label="Seed build"),
                        Line2D([],[],ls="",marker="D",ms=4,mec="#392915",mfc="#ffda8d",label="Follow-up build")],
               loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(.5,-.005))
    save(fig, "figF_probability_map")


def desirability(field):
    """Equation (11). Porosity is the predictive median, expressed in percent."""
    density = np.clip((field["rho"][0] - 8.20) / (8.44 - 8.20), 0, 1)
    hardness = np.clip((field["HV"][0] - 270) / (285 - 270), 0, 1)
    porosity = np.clip((0.90 - 10 ** field["logp"][0]) / (0.90 - 0.05), 0, 1)
    return np.cbrt(density * hardness * porosity)


def desirability_figure(seeds, model):
    hatch = 0.080
    speed = np.linspace(600, 1400, 181)
    power = np.linspace(150, 400, 181)
    v, p = np.meshgrid(speed, power)
    plane = g.features(p.ravel(), v.ravel(), np.full(v.size, hatch))
    scores = desirability(predictions(model, plane))
    best = np.argmax(scores)
    candidates = g.features(*g.candidates())
    candidate_scores = desirability(predictions(model, candidates))
    best_candidate = np.argmax(candidate_scores)
    best_cei = np.argmax(g.acquisition(model, seeds, candidates)[0])
    rows = []
    for domain, points, values, index in (
        ("h = 0.080 mm plane", plane, scores, best),
        ("maximum D among 10000 candidates", candidates, candidate_scores, best_candidate),
        ("maximum CEI among 10000 candidates", candidates, candidate_scores, best_cei),
    ):
        rows.append([domain, len(seeds), index, *points[index, :3], values[index]])
    pd.DataFrame(rows, columns=["domain", "training_builds", "index", "P", "v", "h", "D"]).to_csv(
        "desirability_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    fig.subplots_adjust(left=.12, right=.88, bottom=.24, top=.91)
    levels = np.linspace(0, scores[best], 22)
    surface = ax.contourf(speed, power, scores.reshape(v.shape), levels=levels, cmap="viridis")
    cb = fig.colorbar(surface, ax=ax, pad=.028, ticks=np.linspace(0, scores[best], 8), format="%.4f")
    cb.set_label("Desirability")
    outline = seeds.loc[np.isclose(seeds.h, hatch), ["v", "P"]].to_numpy()
    vertices = ConvexHull(outline).vertices
    boundary = outline[np.r_[vertices, vertices[0]]]
    ax.plot(speed, 85 * speed * hatch * g.T_LAYER, "--", color="white", lw=1)
    ax.plot(boundary[:, 0], boundary[:, 1], color="#d65c13", lw=1.35)
    ax.scatter(outline[:, 0], outline[:, 1], s=14, facecolors="white", edgecolors="#26333d", lw=.7)
    ax.scatter(plane[best, 1], plane[best, 0], marker="*", s=105,
               facecolors="#e98c30", edgecolors="black", lw=.7, zorder=5)
    ax.set(xlim=(600, 1400), ylim=(150, 400), xlabel="Scan speed (mm/s)", ylabel="Laser power (W)",
           title="Seven seed builds, h = 0.080 mm")
    fig.legend(handles=[Line2D([], [], color="#555555", ls="--", label="VED = 85 J/mm³"),
                        Line2D([], [], color="#d65c13", label="Seed hull at this hatch"),
                        Line2D([], [], ls="", marker="*", ms=9, mfc="#e98c30",
                               mec="black", label="Grid maximum")],
               loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(.48,.015))
    save(fig, "figD_desirability")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-trees", action="store_true", help="Also refit the RF and XGBoost reference models.")
    args = parser.parse_args()
    mpl.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "pdf.fonttype":42, "svg.fonttype":"none"})
    builds = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "builds.csv")
    seeds = builds.loc[builds.campaign=="seed"].reset_index(drop=True)
    fitted = {"Hybrid-GPR":g.fit_all(seeds, True), "Zero-mean GP":g.fit_all(seeds, False)}
    evaluate_box(builds, seeds, fitted, args.with_trees)
    slice_figures(builds, seeds, fitted)
    desirability_figure(seeds, fitted["Hybrid-GPR"])
