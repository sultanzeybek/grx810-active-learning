"""Active-learning replay on a common test set (Sect. 4.8, Fig. 15).

Run from results/: python ../code/active_learning_replay.py
"""
from functools import partial
from itertools import combinations, permutations
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


TARGETS = ("rho", "HV", "logp")
POLICIES = ("Random", "Maximum uncertainty", "Cost-aware CEI")
TEST_BUILDS = ("A", "C", "D")


def inputs(frame):
    return g.features(frame.P, frame.v, frame.h)


def fit_subset(training, variant):
    if variant == "fixed":
        return g.fit_all(training)
    points = inputs(training)
    priors = {
        "rho": partial(g.prior_rho, rho_min=training.rho.min()),
        "HV": partial(g.prior_hv, H0=training.HV.min(), dH=np.ptp(training.HV),
                      cref=np.median(points[:, 4])),
        "logp": partial(g.prior_logp, a0=np.log10(training.p2D.median())),
    }
    return {target: g.PhysicsGP(priors[target]).fit(
        points, g.target_values(training, target), g.noise(training, target)
    ) for target in TARGETS}


def best_index(scores):
    """Break numerical ties at floating-point precision by the lower seed ID."""
    best = np.max(scores)
    tol = 32 * np.spacing(abs(best))
    return np.flatnonzero(best - scores <= tol)[0]


def fit_states(pool, test, variant):
    # Canonical training order makes the same subset yield the same fitted model.
    # Follow-up outcomes are used only to calculate the three test errors.
    labels = tuple(pool.index)
    test_x = inputs(test)
    errors, next_choice = {}, {}
    for size in range(3, len(labels) + 1):
        for state in combinations(labels, size):
            training = pool.loc[list(state)]
            fitted = fit_subset(training, variant)
            errors[state] = np.array([np.abs(
                fitted[target].predict(test_x)[0] - g.target_values(test, target)
            ).mean() for target in TARGETS])
            remaining = [label for label in labels if label not in state]
            if remaining:
                candidates = inputs(pool.loc[remaining])
                density_sd = fitted["rho"].predict(candidates)[1]
                cei = g.acquisition(fitted, training, candidates)[0]
                next_choice[state, "Maximum uncertainty"] = remaining[best_index(density_sd)]
                next_choice[state, "Cost-aware CEI"] = remaining[best_index(cei)]
    return errors, next_choice


def replay(pool, test, variant):
    errors, next_choice = fit_states(pool, test, variant)
    records = []
    for start in combinations(pool.index, 3):
        remaining = [label for label in pool.index if label not in start]
        orders = list(permutations(remaining))
        for policy in POLICIES:
            state = start
            for size in range(len(start), len(pool) + 1):
                if policy == "Random":
                    added = size - len(start)
                    mae = np.mean([errors[tuple(sorted(start + order[:added]))]
                                   for order in orders], axis=0)
                else:
                    mae = errors[state]
                picked = ""
                if policy != "Random" and size < len(pool):
                    picked = next_choice[state, policy]
                    state = tuple(sorted(state + (picked,)))
                records.append({"prior": variant, "start": "/".join(start), "n": size,
                                "policy": policy, "next_seed": picked, **dict(zip(TARGETS, mae))})
    return pd.DataFrame(records)


def plot_replay(summary, directory):
    mpl.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    styles = [("#4e5561", "o", "-"), ("#147b83", "s", "--"), ("#b74725", "^", "-.")]
    panels = [("rho", "Density", "MAE (g/cm³)"), ("HV", "Hardness", "MAE (HV1)"),
              ("logp", "Optical porosity", "MAE (decades)")]
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.75))
    for ax, (target, heading, unit) in zip(axes, panels):
        for policy, (colour, marker, linestyle) in zip(POLICIES, styles):
            series = summary[(summary.prior == "fixed") & (summary.policy == policy)]
            ax.plot(series.n, series[target], label=policy, color=colour, marker=marker,
                    linewidth=1.6, markersize=5, linestyle=linestyle,
                    markerfacecolor="white", markeredgewidth=1.05)
        ax.set(title=heading, ylabel=unit, xlabel="Seed builds in training", xticks=range(3, 8))
        ax.grid(axis="y", color="#dadde0", linewidth=.65)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(.5, -.005))
    fig.subplots_adjust(left=.081, right=.987, bottom=.32, top=.88, wspace=.48)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(directory / f"figAL_common_test.{suffix}", dpi=360, bbox_inches="tight",
                    pad_inches=.055)
    plt.close(fig)


def report_feature_check(pool):
    points = inputs(pool)
    for target in TARGETS:
        observations = g.target_values(pool, target)
        noise = g.noise(pool, target)
        maes = []
        for num_inputs in (3, 5):
            errors = []
            for i in range(len(pool)):
                train = np.arange(len(pool)) != i
                model = g.PhysicsGP(g.zero).fit(points[train, :num_inputs],
                                                observations[train], noise[train])
                prediction = model.predict(points[[i], :num_inputs])[0][0]
                errors.append(abs(prediction - observations[i]))
            maes.append(np.mean(errors))
        print(f"{target}: three-input MAE {maes[0]:.6f}, five-input MAE {maes[1]:.6f}, "
              f"relative change {100 * (maes[1] / maes[0] - 1):+.4f}%")


def nested_prior_errors(pool):
    """Re-estimate the seed-informed priors in each of seven leave-one-out folds."""
    rows = []
    for build in pool.index:
        held_out = pool.loc[[build]]
        models = fit_subset(pool.drop(build), "subset")
        row = {"held_out": build}
        for target in TARGETS:
            predicted = models[target].predict(inputs(held_out))[0][0]
            observed = g.target_values(held_out, target)[0]
            row[target] = abs(predicted - observed)
        rows.append(row)
    return pd.DataFrame(rows).set_index("held_out")


def main():
    directory = Path.cwd()
    data = pd.read_csv(_builds_csv()).set_index("build")
    pool = data[data.campaign == "seed"].sort_index()
    test = data.loc[list(TEST_BUILDS)]
    detail = pd.concat([replay(pool, test, v) for v in ("fixed", "subset")], ignore_index=True)
    summary = detail.groupby(["prior", "n", "policy"], sort=False, as_index=False)[list(TARGETS)].mean()
    detail.to_csv(directory / "replay_by_start.csv", index=False, float_format="%.10f")
    summary.to_csv(directory / "replay_summary.csv", index=False, float_format="%.10f")
    plot_replay(summary, directory)
    for variant in ("fixed", "subset"):
        middle = summary[(summary.prior == variant) & summary.n.between(4, 6)]
        random = middle[middle.policy == "Random"].set_index("n")[list(TARGETS)]
        cei = middle[middle.policy == "Cost-aware CEI"].set_index("n")[list(TARGETS)]
        print(f"{variant} priors: mean MAE of CEI minus exact random (A, C and D)")
        print((cei - random).to_string(float_format=lambda x: f"{x:.6f}"))
    print("Figure 15 uses the fixed priors. Each policy has the same 35 starts.")
    nested = nested_prior_errors(pool)
    nested.to_csv(directory / "nested_prior_loocv.csv")
    print("Training-fold priors, seven-fold MAE:")
    print(nested.mean().to_string(float_format=lambda value: f"{value:.6f}"))
    report_feature_check(pool)
    grid = g.features(*g.candidates())
    seed_model = g.fit_all(pool)
    grid_cei = g.acquisition(seed_model, pool, grid)[0]
    c_cei, c_probability, _, _, fstar = g.acquisition(seed_model, pool, inputs(test.loc[["C"]]))
    print(f"Seven-build f* {fstar:.6f}, C CEI/maximum {c_cei[0]/grid_cei.max():.6f}, "
          f"grid candidates above C {np.sum(grid_cei > c_cei[0])}/{len(grid)}, "
          f"joint latent probability at C {c_probability[0]:.6f}")
    fstar_updated = g.acquisition(g.fit_all(data), data, inputs(test.loc[["C"]]))[-1]
    print(f"Eleven-build f* {fstar_updated:.6f}")


if __name__ == "__main__":
    main()
