"""
Reproduce the key numerical results reported in the paper.

Run:  python src/reproduce_analyses.py

Outputs the LOOCV tournament, the feature ablation, the Hybrid-vs-vanilla
comparison, the feasibility coverage, the cross-laboratory check, the
VED-porosity correlation, the cost-coefficient sensitivity, and the
desirability single-point recommendation.
"""

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from scipy.stats import wilcoxon, pearsonr

from grx810_framework import (
    load_data, make_gpr, Hybrid, cost_aware_cei, pareto_front, bootstrap_ci,
    prior_density, prior_hardness, prior_logp, LAYER_THICKNESS, FEATURES)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
t = LAYER_THICKNESS


def loocv_mae(make_model, X, y):
    err = []
    for tr, te in LeaveOneOut().split(X):
        err.append(abs(y[te][0] - make_model(X[tr], y[tr], X[te])[0]))
    return np.array(err)


def fit_pred(kind, prior=None):
    def f(Xtr, ytr, Xte):
        if kind == "hybrid":
            return Hybrid(prior).fit(Xtr, ytr).predict(Xte)
        if kind == "gpr":
            sc = StandardScaler().fit(Xtr)
            return make_gpr().fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))
        if kind == "rf":
            return RandomForestRegressor(50, max_depth=3, random_state=0).fit(Xtr, ytr).predict(Xte)
        if kind == "svr":
            sc = StandardScaler().fit(Xtr)
            return SVR(C=10, epsilon=0.1).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))
        if kind == "xgb":
            return XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1,
                                verbosity=0).fit(Xtr, ytr).predict(Xte)
    return f


def main():
    df, X = load_data(os.path.join(HERE, "data", "builds.csv"))
    targets = {"density": df.rho.values, "hardness": df.HV.values, "log10_p2D": df.logp.values}

    print("=" * 60)
    print("1. LOOCV tournament (MAE, 95% bootstrap CI)")
    print("=" * 60)
    for name, y in targets.items():
        print(f"\n[{name}]")
        for kind in ["rf", "svr", "gpr", "xgb"]:
            e = loocv_mae(fit_pred(kind), X, y)
            lo, hi = bootstrap_ci(e)
            print(f"  {kind:5s} MAE={e.mean():.3f}  CI=[{lo:.3f},{hi:.3f}]")
        prior = {"density": prior_density, "hardness": prior_hardness, "log10_p2D": prior_logp}[name]
        e = loocv_mae(fit_pred("hybrid", prior), X, y)
        lo, hi = bootstrap_ci(e)
        print(f"  hybrid MAE={e.mean():.3f}  CI=[{lo:.3f},{hi:.3f}]")

    print("\n" + "=" * 60)
    print("2. Feature ablation (vanilla GPR, nested feature sets)")
    print("=" * 60)
    sets = {"raw{P,v,h}": [0, 1, 2], "raw+VED": [0, 1, 2, 3], "raw+all": [0, 1, 2, 3, 4, 5]}
    for name, y in targets.items():
        vals = []
        for cols in sets.values():
            e = loocv_mae(fit_pred("gpr"), X[:, cols], y)
            vals.append(e.mean())
        d = 100 * (vals[-1] - vals[0]) / vals[0]
        print(f"  {name:10s} raw->all change = {d:+.2f}%")

    print("\n" + "=" * 60)
    print("3. Hybrid vs vanilla GPR (Wilcoxon, one-sided)")
    print("=" * 60)
    for name, y in targets.items():
        prior = {"density": prior_density, "hardness": prior_hardness, "log10_p2D": prior_logp}[name]
        eg = loocv_mae(fit_pred("gpr"), X, y)
        eh = loocv_mae(fit_pred("hybrid", prior), X, y)
        try:
            p = wilcoxon(eh, eg, alternative="less").pvalue
        except ValueError:
            p = float("nan")
        print(f"  {name:10s} GPR={eg.mean():.3f} Hybrid={eh.mean():.3f} "
              f"delta={100*(eh.mean()-eg.mean())/eg.mean():+.0f}% p={p:.3f}")

    print("\n" + "=" * 60)
    print("4. VED-porosity correlation")
    print("=" * 60)
    r1, p1 = pearsonr(df.VED.values, df.p2D.values)
    r2, _ = pearsonr(np.abs(df.VED.values - 85), df.logp.values)
    print(f"  Pearson VED vs p2D       : r={r1:+.3f} p={p1:.3f}")
    print(f"  Pearson |VED-85| vs logp : r={r2:+.3f}")

    print("\n" + "=" * 60)
    print("5. Cross-laboratory check (full-fit posterior at S2 centreline)")
    print("=" * 60)
    m = Hybrid(prior_density).fit(X, df.rho.values)
    pred = m.predict(X[1:2])[0]
    print(f"  posterior={pred:.3f}  M290 obs={df.rho.values[1]:.3f}  "
          f"NASA M280=8.440  rel.err={100*abs(pred-8.440)/8.440:.3f}%")

    print("\n" + "=" * 60)
    print("6. Cost-coefficient sensitivity")
    print("=" * 60)
    rng = np.random.default_rng(7)
    n = 10000
    P = rng.uniform(150, 400, n); v = rng.uniform(600, 1400, n); h = rng.uniform(0.06, 0.10, n)
    VED = P / (v * h * t); Cd = 1000 * np.sqrt(v) / P; TI = P / np.sqrt(v)
    G = np.column_stack([P, v, h, VED, Cd, TI])
    hr = Hybrid(prior_density).fit(X, df.rho.values)
    hh = Hybrid(prior_hardness).fit(X, df.HV.values)
    mu_d, sd_d = hr.predict(G, return_std=True); hv = hh.predict(G)
    feas = (mu_d >= 8.39) & (hv >= 275) & (VED < 100)
    for lam in [0.0, 0.001, 0.005, 0.05]:
        a = cost_aware_cei(mu_d, sd_d, df.rho.values.max(), feas, P, lam=lam)
        i = int(np.argmax(a))
        print(f"  lambda={lam:.3f}  P*={P[i]:.0f}W  VED*={VED[i]:.1f}")

    print("\n" + "=" * 60)
    print("7. Desirability single-point recommendation")
    print("=" * 60)
    rng = np.random.default_rng(42)
    P = rng.uniform(150, 400, 3000); v = rng.uniform(600, 1400, 3000); h = rng.uniform(0.06, 0.10, 3000)
    VED = P / (v * h * t); Cd = 1000 * np.sqrt(v) / P; TI = P / np.sqrt(v)
    G = np.column_stack([P, v, h, VED, Cd, TI])
    ro = hr.predict(G); ho = hh.predict(G); p2d = 10 ** Hybrid(prior_logp).fit(X, df.logp.values).predict(G)
    mask = (p2d < 0.2) & (ho >= 275) & (VED < 100)
    i = np.where(mask)[0][np.argmax(ro[mask])]
    print(f"  rule: max density s.t. p2D<0.2%, HV>=275")
    print(f"  -> P={P[i]:.0f}W v={v[i]:.0f} h={h[i]:.3f} VED={VED[i]:.1f} "
          f"rho={ro[i]:.3f} HV={ho[i]:.0f} p2D={p2d[i]:.3f}%")


if __name__ == "__main__":
    main()
