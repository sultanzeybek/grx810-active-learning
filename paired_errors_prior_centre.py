"""Paired LOOCV errors (Table 7, Fig. 7) and the prior-centre comparison (Table A2, Fig. 8).

Run from results/: python ../code/paired_errors_prior_centre.py
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import nct, norm, t, wilcoxon

import hybrid_gpr as g

CENTRES = tuple(range(70, 111, 5))
TARGETS = ("rho", "HV", "logp")
STYLES = (("rho", "Density", "#204f75", "o"), ("HV", "Hardness", "#a44c16", "s"),
          ("logp", "Optical porosity (log scale)", "#397a66", "^"))


def features(frame):
    return g.features(frame.P, frame.v, frame.h)


def paired_errors(seed):
    rows = []
    for name, group in seed.groupby("build", sort=False):
        train = seed.loc[seed.build != name]
        models = {"zero": g.fit_all(train, False), "hybrid": g.fit_all(train, True)}
        for target in TARGETS:
            observed = g.target_values(group, target)[0]
            row = {"held_out": name, "target": target, "observed": observed,
                   "noise_sd": g.noise(group, target)[0]}
            for label, fit in models.items():
                prediction = fit[target].predict(features(group))[0][0]
                row[label + "_prediction"] = prediction
                row[label + "_abs_error"] = abs(prediction - observed)
            row["difference"] = row["zero_abs_error"] - row["hybrid_abs_error"]
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(folds):
    builds = folds.held_out.unique()
    draw = np.random.default_rng(0).integers(len(builds), size=(1000, len(builds)))
    rows = []
    for target, losses in folds.groupby("target", sort=False):
        losses = losses.set_index("held_out").loc[builds]
        difference = losses.difference
        s = losses.noise_sd.iloc[0]
        if not np.allclose(losses.noise_sd, s):
            raise ValueError("A common noise SD is required for each seed target.")
        row = {"target": target, "noise_mae": s * np.sqrt(2 / np.pi),
               "delta_mae": difference.mean(), "delta_over_s": difference.mean() / s,
               "paired_dz": difference.mean() / difference.std(ddof=1),
               "hybrid_wins": (difference > 0).sum(),
               "nominal_p": wilcoxon(difference, alternative="greater", method="exact").pvalue}
        for model in ("zero", "hybrid"):
            error = losses[model + "_abs_error"].to_numpy()
            row[model + "_mae"] = error.mean()
            row[model + "_lo"], row[model + "_hi"] = np.quantile(error[draw].mean(axis=1), [.025, .975])
        rows.append(row)
    return pd.DataFrame(rows).set_index("target")


def planning(dz):
    """Minimum integer sample sizes in the prospective designs in Sects. 4.2 and 5.2."""
    n = 2
    while nct.sf(t.isf(.05, n-1), n-1, dz * np.sqrt(n)) < .80:
        n += 1
    print(f"Density: normal approximation {((norm.isf(.05) + norm.ppf(.80))/dz)**2:.3f}; paired t: {n}.")
    for k in (5, 10):
        sd = g.S_HV1 / np.sqrt(k)
        n = 2
        while True:
            df = 2*n - 2
            critical = t.isf(.025, df)
            nc = 5.5 * np.sqrt(n/2) / sd
            if nct.sf(critical, df, nc) + nct.cdf(-critical, df, nc) >= .80:
                break
            n += 1
        print(f"Hardness: {n} independent builds per setting, {k} indents per build.")


def fit_density(frame, centre, ceiling=8.44):
    model = g.PhysicsGP(lambda x: g.prior_rho(x, ved_opt=centre, rho_max=ceiling))
    return model.fit(features(frame), frame.rho.to_numpy(), g.noise(frame, "rho"))


def centre_errors(frame, centre):
    out = []
    for i in range(len(frame)):
        train, test = frame.drop(frame.index[i]), frame.iloc[[i]]
        prediction = fit_density(train, centre).predict(features(test))[0][0]
        out.append(abs(prediction - test.rho.iloc[0]))
    return np.asarray(out)


def prior_sweep(data, seed):
    averaged = seed.copy()
    averaged.loc[averaged.build == "S2", "rho"] = data.loc[data.build.isin(["S2", "B"]), "rho"].mean()
    cv = []
    for name, frame in (("seeds", seed), ("S2_B_mean", averaged), ("all_eleven", data)):
        reference = centre_errors(frame, 85)
        for centre in CENTRES:
            errors = reference if centre == 85 else centre_errors(frame, centre)
            diff = errors - reference
            p = np.nan if centre == 85 else wilcoxon(diff, alternative="greater", method="exact").pvalue
            cv.append({"dataset": name, "centre": centre, "density_mae": errors.mean(),
                       "nominal_p_vs_85": p, "bonferroni_p": np.minimum(1., p * (len(CENTRES)-1)),
                       "folds_favouring_85": (diff > 0).sum() if centre != 85 else np.nan})
    power, speed, hatch = g.candidates()
    candidates = g.features(power, speed, hatch)
    hull = g.in_hull(seed, power, speed, hatch)
    hardness = g.fit_all(seed)["HV"].predict(candidates)[0]
    prior_region = (g.prior_rho(candidates) >= g.RHO_THR) & (g.prior_hv(candidates) >= g.HV_THR)
    regions = []
    for ceiling in (8.44, 8.50):
        for centre in CENTRES:
            density = fit_density(seed, centre, ceiling).predict(candidates)[0]
            region = (density >= g.RHO_THR) & (hardness >= g.HV_THR)
            regions.append({"ceiling": ceiling, "centre": centre, "count": region.sum(),
                            "centroid": candidates[region,3].mean() if region.any() else np.nan})
            if ceiling == 8.44 and centre == 85:
                print(f"Full-box VED centroids: prior {candidates[prior_region,3].mean():.5f}, "
                      f"posterior {candidates[region,3].mean():.5f}. In-hull counts: prior "
                      f"{(prior_region & hull).sum()}, posterior {(region & hull).sum()}, "
                      f"intersection {(region & prior_region & hull).sum()} of {hull.sum()}.")
    return pd.DataFrame(cv), pd.DataFrame(regions)


def figure7(seed, folds, summary):
    # The three non-GP fitted baselines are the means in Table 7. The reference
    # predictors and both GPs are calculated from the builds in this package.
    models = ["Hybrid-GPR", "Zero-mean GP", "Prior only", "Seed mean", "Random forest", "SVR", "XGBoost"]
    means = pd.DataFrame(index=models, columns=TARGETS, dtype=float)
    means.loc["Hybrid-GPR"] = summary.hybrid_mae
    means.loc["Zero-mean GP"] = summary.zero_mae
    for target in TARGETS:
        y = g.target_values(seed, target)
        means.loc["Prior only",target] = np.abs(y - g.PRIORS[target](features(seed))).mean()
        means.loc["Seed mean",target] = np.abs(y - (y.sum()-y)/(len(y)-1)).mean()
    means.loc[["Random forest","SVR","XGBoost"]] = [[.111,6.81,.352],[.106,6.05,.329],[.136,5.83,.465]]
    builds = seed.build.to_list()
    fig,(left,right) = plt.subplots(1,2,figsize=(8.4,4),gridspec_kw={"width_ratios":[1.08,1]})
    fig.subplots_adjust(left=.155,right=.973,top=.78,bottom=.20,wspace=.34)
    for axis in (left,right):
        axis.set_axisbelow(True)
        axis.grid(axis="x",color="#e6e9eb",linewidth=.65)
        axis.spines[["top","right","left"]].set_visible(False)
        axis.spines["bottom"].set_color("#a0a6ab")
        axis.tick_params(length=0,pad=4)
    for offset,(target,label,colour,marker) in zip((-.22,0,.22),STYLES):
        errors = folds.loc[folds.target==target].set_index("held_out").loc[builds]
        s=errors.noise_sd.iloc[0]
        args = dict(marker=marker,color=colour,edgecolors="white",linewidths=.35,zorder=3)
        left.scatter(means[target]/s,np.arange(len(models))+offset,s=23,label=label,**args)
        difference=errors.difference/s
        right.scatter(difference,np.arange(len(builds))+offset,s=23,**args)
        right.scatter(difference.mean(),len(builds)+.4+offset,s=31,**args)
    left.set_yticks(range(len(models)),models)
    left.set(xlim=(0,2.8),ylim=(len(models)-.35,-.65),xlabel="Mean absolute error / noise SD")
    left.axvline(np.sqrt(2/np.pi),color="#48515a",ls="--",lw=1)
    left.set_title("(a)  Error on the seven seed builds",loc="left",fontsize=10,pad=9)
    right.set_yticks([*range(len(builds)),len(builds)+.4],[*builds,"Mean"])
    right.set(xlim=(-1.2,1.2),ylim=(len(builds)+1.05,-.65),xlabel="(Zero-mean GP error − Hybrid-GPR error) / noise SD")
    right.set_xticks([-1,-.5,0,.5,1])
    right.axvline(0,color="#65717a",lw=1)
    right.axhline(len(builds)-.3,color="#cdd2d5",lw=.8)
    right.set_title("(b)  Paired difference by held-out build",loc="left",fontsize=10,pad=9)
    fig.legend(*left.get_legend_handles_labels(),loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(.53,.99))
    return fig


def figure8(cv,regions,zero_mae):
    fig,(left,right)=plt.subplots(1,2,figsize=(8.1,3.65))
    fig.subplots_adjust(left=.09,right=.98,top=.86,bottom=.24,wspace=.30)
    left.plot([68,113],[68,113],":",color="#697078",linewidth=1,label="Centroid = prior centre")
    for ceiling,marker,colour in ((8.44,"o","#204f75"), (8.50,"s","#5593bf")):
        sub=regions.loc[regions.ceiling==ceiling]
        left.plot(sub.centre,sub.centroid,marker=marker,color=colour,ms=4,lw=1.35,label=rf"$\rho_{{\max}}={ceiling:.2f}$")
    left.set(xlim=(67,113),ylim=(63,114),ylabel="Feasible-region centroid (J/mm³)")
    left.set_title("(a)  Location of the feasible region",loc="left",fontsize=10,pad=9)
    sel=cv.loc[cv.dataset=="seeds"]
    right.plot(sel.centre,sel.density_mae,"o-",color="#204f75",ms=4,lw=1.35)
    right.axhline(zero_mae,ls="--",color="#697078",lw=1)
    right.annotate("Zero-mean GP",(110,zero_mae),xytext=(0,4),textcoords="offset points",ha="right",size=7)
    right.set(xlim=(68,112),ylabel="Density LOOCV MAE (g/cm³)")
    right.set_title("(b)  Error on the seven seed builds",loc="left",fontsize=10,pad=9)
    for axis in (left,right):
        axis.set_xlabel("Prior centre (J/mm³)")
        axis.set_xticks([70,80,90,100,110])
        axis.spines[["top","right"]].set_visible(False)
        axis.tick_params(length=3,width=.7)
    handles,labels=left.get_legend_handles_labels()
    fig.legend([handles[i] for i in (1,2,0)],[labels[i] for i in (1,2,0)],loc="lower center",
               ncol=3,frameon=False,bbox_to_anchor=(.5,-.01))
    return fig


if __name__ == "__main__":
    mpl.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42,"svg.fonttype":"none"})
    data = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "builds.csv")
    seed = data.loc[data.campaign=="seed"].reset_index(drop=True)
    folds = paired_errors(seed)
    folds.to_csv("loocv_paired_errors.csv",index=False)
    summary = summarise(folds)
    summary.to_csv("loocv_paired_summary.csv")
    print(summary.round(5).to_string())
    planning(summary.loc["rho","paired_dz"])
    cv,regions=prior_sweep(data,seed)
    cv.to_csv("prior_centre_loocv.csv",index=False)
    regions.to_csv("prior_centre_regions.csv",index=False)
    for fig,name in [(figure7(seed,folds,summary),"figLOO_noise_floor"),(figure8(cv,regions,summary.loc["rho","zero_mae"]),"figP_prior_sensitivity")]:
        for fmt in ("pdf","svg","png"):
            fig.savefig(f"{name}.{fmt}",dpi=360,bbox_inches="tight",pad_inches=.06)
        plt.close(fig)
