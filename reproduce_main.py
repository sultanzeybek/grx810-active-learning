"""Reproduce the main Gaussian-process results of the paper from data/builds.csv.

Run from results/:  python ../code/reproduce_main.py
Covers: LOOCV (Table 7), feasible region (Table 8), follow-up test and model
update (Table 10), seed vs follow-up comparison, sample-size arithmetic and the
value of a twelfth build under different density noise.
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import hybrid_gpr as g


def loocv(seed):
    out = {}
    for name, phys in (("Hybrid-GPR", True), ("Zero-mean GP", False)):
        err = {t: [] for t in ("rho", "HV", "logp")}
        for i in range(len(seed)):
            tr, te = seed.drop(seed.index[i]), seed.iloc[[i]]
            m = g.fit_all(tr, phys)
            Xt = g.features(te.P, te.v, te.h)
            for t in err:
                err[t].append(abs(m[t].predict(Xt)[0][0] - g.target_values(te, t)[0]))
        out[name] = {t: np.mean(e) for t, e in err.items()}
    X = g.features(seed.P, seed.v, seed.h)
    out["Prior only"] = {"rho": np.mean(abs(g.prior_rho(X) - seed.rho)),
                         "HV": np.mean(abs(g.prior_hv(X) - seed.HV)),
                         "logp": np.mean(abs(g.prior_logp(X) - np.log10(seed.p2D)))}
    return pd.DataFrame(out).T


def rho_min_sensitivity(seed):
    """Density LOOCV MAE for four rules that set the lower asymptote rho_min (Sect. 4.2)."""
    rules = {"fixed 8.20 (main model)": lambda tr: 8.20,
             "fixed 0.97 x 8.44": lambda tr: 0.97 * 8.44,
             "fold minimum (nested)": lambda tr: tr.rho.min(),
             "fold minimum - 0.05": lambda tr: tr.rho.min() - 0.05}
    for name, rule in rules.items():
        err = []
        for i in range(len(seed)):
            tr, te = seed.drop(seed.index[i]), seed.iloc[[i]]
            rmin = rule(tr)
            m = g.PhysicsGP(lambda X, r=rmin: g.prior_rho(X, rho_min=r)).fit(
                g.features(tr.P, tr.v, tr.h), tr.rho.values, g.noise(tr, "rho"))
            err.append(abs(m.predict(g.features(te.P, te.v, te.h))[0][0] - te.rho.values[0]))
        print(f"{name:26s} density MAE {np.mean(err):.5f}")


def hyper_per_fold(seed):
    rows = []
    for name, phys in (("Zero-mean GP", False), ("Hybrid-GPR", True)):
        for i in range(len(seed)):
            ms = g.fit_all(seed.drop(seed.index[i]), phys)
            for t in ("rho", "HV", "logp"):
                rows.append((name, seed.build.iloc[i], t, ms[t].ell, ms[t].sf2))
    df = pd.DataFrame(rows, columns=["model", "held_out", "target", "ell", "sigma_f2"])
    df.to_csv("hyperparameters_per_fold.csv", index=False)
    print(df.groupby(["target", "model"])[["ell", "sigma_f2"]].agg(["min", "max"]).round(4))


def window(df, Xc, hull):
    m, m0 = g.fit_all(df, True), g.fit_all(df, False)
    a, pf, mr, mh, fstar = g.acquisition(m, df, Xc)
    _, pf0, mr0, mh0, _ = g.acquisition(m0, df, Xc)
    reg = (mr >= g.RHO_THR) & (mh >= g.HV_THR)
    reg0 = (mr0 >= g.RHO_THR) & (mh0 >= g.HV_THR)
    pr = (g.prior_rho(Xc) >= g.RHO_THR) & (g.prior_hv(Xc) >= g.HV_THR)
    pp = norm.cdf((g.prior_rho(Xc) - g.RHO_THR) / g.S_RHO) * \
        norm.cdf((g.prior_hv(Xc) - g.HV_THR) / (g.S_HV1 / np.sqrt(5)))
    i = a.argmax()
    print(f"f* = {fstar:.3f}, beta = {m['rho'].beta:.3f}; top candidate "
          f"{Xc[i,0]:.0f} W / {Xc[i,1]:.0f} mm/s / {Xc[i,2]:.3f} mm (VED {Xc[i,3]:.1f}), Pr_feas {pf[i]:.2f}")
    print(f"{'model':22s}{'mean %':>8s}{'hull n':>8s}{'max latent Pr':>15s}")
    print(f"{'Hybrid-GPR':22s}{100*reg.mean():8.2f}{(reg&hull).sum():8d}{pf.max():15.3f}")
    print(f"{'Zero-mean GP':22s}{100*reg0.mean():8.2f}{(reg0&hull).sum():8d}{pf0.max():15.3f}")
    print(f"{'Prior only':22s}{100*pr.mean():8.2f}{(pr&hull).sum():8d}{'n/a':>15s}")
    print(f"Prior only, peak probability for a future measured build with five indents: {pp.max():.3f}")
    print(f"GP region inside prior region: {np.all(pr[reg])}; "
          f"IoU(GP, prior) hull {(reg&pr&hull).sum()/((reg|pr)&hull).sum():.2f}, box {(reg&pr).sum()/(reg|pr).sum():.2f}")


def update(d, Xc):
    s, f = d[d.campaign == "seed"], d[d.campaign == "followup"]
    for tag, df in (("N = 7", s), ("N = 8 (+A, analytic subset)", pd.concat([s, f[f.build == "A"]])),
                    ("N = 11", d)):
        m = g.fit_all(df)
        a, pf, mr, mh, _ = g.acquisition(m, df, Xc)
        i = a.argmax()
        print(f"{tag:30s} max rho {mr.max():.3f}  max Pr {pf.max():.2f}  region "
              f"{100*((mr>=g.RHO_THR)&(mh>=g.HV_THR)).mean():4.1f}%  top VED {Xc[i,3]:.1f}  beta {m['rho'].beta:+.3f}")
    m7 = g.fit_all(s)
    X = g.features(f.P, f.v, f.h)
    for t in ("rho", "HV", "logp"):
        mu, sd = m7[t].predict(X)
        y, sn = g.target_values(f, t), g.noise(f, t)
        print(f"  test {t:5s} MAE {np.mean(abs(mu-y)):.3f}  z = {np.round((y-mu)/np.sqrt(sd**2+sn**2), 2)}")


def seed_vs_followup(d):
    s, f = d[d.campaign == "seed"], d[d.campaign == "followup"]
    ok = lambda x: ((x.rho >= g.RHO_THR) & (x.HV >= g.HV_THR)).sum()
    r = d.rho.values
    obs = f.rho.mean() - s.rho.mean()
    diffs = []
    for idx in itertools.combinations(range(len(d)), len(f)):
        m = np.zeros(len(d), bool); m[list(idx)] = True
        diffs.append(r[m].mean() - r[~m].mean())
    p = np.mean(np.abs(diffs) >= abs(obs) - 1e-12)
    print(f"density mean seed {s.rho.mean():.3f} (SD {s.rho.std():.3f}), follow-up {f.rho.mean():.3f} "
          f"(SD {f.rho.std():.3f}); difference {obs:+.3f}, exact permutation p = {p:.2f}")
    print(f"both constraints met: seed {ok(s)}/{len(s)}, follow-up {ok(f)}/{len(f)}")
    print(f"S2 vs its replicate B: {d.rho[d.build=='S2'].item():.3f} vs {d.rho[d.build=='B'].item():.3f}")
    resid = np.sqrt(max(r.var(ddof=1) - g.S_RHO ** 2, 0))
    print(f"SD of all eleven densities {r.std(ddof=1):.3f}; part not explained by noise {resid:.3f} g/cm3")


def sample_size():
    z = norm.ppf
    for s in (0.08, 0.04, 0.0085):
        n1 = ((z(.95) + z(.8)) * s / 0.025) ** 2
        n2 = 2 * ((z(.975) + z(.8)) * s / 0.03) ** 2
        print(f"s = {s:.4f}: builds to show a 0.025 margin above 8.40: {n1:5.1f}; "
              f"builds per setting to separate two settings 0.03 apart: {n2:5.1f}")


def value_of_twelfth(d, Xc, n_sim=40, seed=1):
    rng = np.random.default_rng(seed)
    xa = g.features([265], [1100], [0.071])
    s0 = g.S_RHO
    for s in (0.08, 0.04, 0.02):
        g.S_RHO = s
        m = g.fit_all(d)
        _, pf_now, _, _, _ = g.acquisition(m, d, Xc)
        mu, sd = m["rho"].predict(xa)
        res = []
        for _ in range(n_sim):
            new = d[d.build == "A"].copy()
            new["build"], new["rho"] = "A2", rng.normal(mu[0], np.sqrt(sd[0] ** 2 + s ** 2))
            dd = pd.concat([d, new])
            _, pf, _, _, _ = g.acquisition(g.fit_all(dd), dd, Xc)
            res.append(pf.max())
        res = np.array(res)
        print(f"s_rho = {s:.2f}: max Pr now {pf_now.max():.2f}; after a replicate of A "
              f"mean {res.mean():.2f}, share of outcomes above 0.5: {np.mean(res > .5):.2f}")
    g.S_RHO = s0


if __name__ == "__main__":
    d = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "builds.csv")
    seed = d[d.campaign == "seed"].reset_index(drop=True)
    P, v, h = g.candidates()
    Xc = g.features(P, v, h)
    hull = g.in_hull(seed, P, v, h)
    print(f"candidates inside the seed hull: {hull.sum()} of {len(P)}\n")
    print("== LOOCV MAE (seven seed builds)"); print(loocv(seed).round(3), "\n")
    print("== Lower asymptote of the density prior (rho_min)"); rho_min_sensitivity(seed); print()
    print("== Hyperparameters over the LOOCV folds (Table A1; all values in hyperparameters_per_fold.csv)"); hyper_per_fold(seed); print()
    print("== Feasible region at N = 7"); window(seed, Xc, hull); print()
    print("== Model update and test on the follow-up builds"); update(d, Xc); print()
    print("== Seed builds vs follow-up builds"); seed_vs_followup(d); print()
    print("== Sample size at a given density noise"); sample_size(); print()
    print("== Value of a twelfth build (replicate of A)"); value_of_twelfth(d, Xc)
