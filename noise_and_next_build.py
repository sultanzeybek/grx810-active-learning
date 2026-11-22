"""Sensitivity to the density noise and the prior width, the density-correction scenario
and the virtual twelfth build.

Run from results/:  python ../code/noise_and_next_build.py
Covers: Table A3 and the statements of Appendix A.4, the air-layer scenario of Sect. 5.3,
and Table A4 with the statements of Sect. 5.4.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import hybrid_gpr as g

CENTRE = g.features([275], [1000], [0.08])          # NASA centreline
P_LIMIT = 370.0                                     # GRX-810 job limit on this machine
N_SIM, SIM_SEED = 40, 1                             # simulated outcomes of a virtual build
N_INDENTS_NEW = 10                                  # indents of a future build (Sect. 5.3)

# Specimen geometries of the micro-CT scans (Sect. 3.3.3), used for S/V in the scenario.
BAR = (4.6, 4.0, 25.0)                              # seed builds, mm
DISC = (15.7, 4.9)                                  # follow-up builds, diameter and height, mm


def _builds_csv():
    """Locate builds.csv next to the script or in ../data (repository layout)."""
    here = Path(__file__).resolve().parent
    for p in (here / "builds.csv", here.parent / "data" / "builds.csv", Path.cwd() / "builds.csv"):
        if p.exists():
            return p
    raise FileNotFoundError("builds.csv not found next to the script or in ../data")


class Settings:
    """Temporarily change the density noise and the density prior."""

    def __init__(self, s_rho=0.08, **prior):
        self.s_rho, self.prior = s_rho, prior

    def __enter__(self):
        self.old = g.S_RHO, g.PRIORS["rho"]
        g.S_RHO = self.s_rho
        g.PRIORS["rho"] = lambda X, kw=self.prior: g.prior_rho(X, **kw)

    def __exit__(self, *exc):
        g.S_RHO, g.PRIORS["rho"] = self.old


def loocv_rho(seed, physics=True):
    err = []
    for i in range(len(seed)):
        tr, te = seed.drop(seed.index[i]), seed.iloc[[i]]
        prior = g.PRIORS["rho"] if physics else g.zero
        m = g.PhysicsGP(prior).fit(g.features(tr.P, tr.v, tr.h), tr.rho.values, g.noise(tr, "rho"))
        err.append(abs(m.predict(g.features(te.P, te.v, te.h))[0][0] - te.rho.values[0]))
    return np.array(err)


def table_a3(seed, follow, Xc):
    print("== Table A3: density noise s_rho (prior width 18) and prior width (s_rho = 0.08)")
    print(f"{'setting':16s}{'LOOCV':>7s}{'region %':>9s}{'max Pr':>8s}{'Pr_c':>6s}{'top VED':>9s}"
          f"{'top P/v/h':>20s}{'test MAE':>9s}{'dMAE':>7s}")
    print("test MAE: density MAE on A, B, C, D; dMAE: zero-mean GP minus Hybrid-GPR LOOCV MAE")
    rows = [dict(s_rho=s) for s in (0.04, 0.06, 0.08, 0.10)] + \
           [dict(s_rho=0.08, sig=w) for w in (12.0, 15.0, 21.0, 24.0)]
    for r in rows:
        s = r.pop("s_rho")
        with Settings(s, **r):
            e_h, e_0 = loocv_rho(seed), loocv_rho(seed, physics=False)
            m = g.fit_all(seed)
            a, pf, mr, mh, _ = g.acquisition(m, seed, Xc)
            pc = g.acquisition(m, seed, CENTRE)[1][0]
            i = a.argmax()
            mu = m["rho"].predict(g.features(follow.P, follow.v, follow.h))[0]
            tag = f"s={s:.2f}" + (f", w={r['sig']:.0f}" if r else "")
            print(f"{tag:16s}{e_h.mean():7.3f}{100*((mr>=g.RHO_THR)&(mh>=g.HV_THR)).mean():9.2f}"
                  f"{pf.max():8.2f}{pc:6.2f}{Xc[i,3]:9.1f}"
                  f"{f'{Xc[i,0]:.0f}/{Xc[i,1]:.0f}/{Xc[i,2]:.3f}':>20s}"
                  f"{np.mean(abs(mu-follow.rho.values)):9.4f}{(e_0-e_h).mean():7.3f}")
    print()


def surface_to_volume():
    a, b, c = BAR
    sv_bar = 2 * (a * b + a * c + b * c) / (a * b * c)
    d, h = DISC
    r = d / 2
    sv_disc = (2 * np.pi * r ** 2 + 2 * np.pi * r * h) / (np.pi * r ** 2 * h)
    return sv_bar, sv_disc


def air_layer_scenario(d, Xc):
    sv_bar, sv_disc = surface_to_volume()
    print("== Sect. 5.3: readings corrected as rho (1 + delta S/V)")
    print(f"S/V bar {sv_bar:.3f} 1/mm, disc {sv_disc:.3f} 1/mm; air layer for a 1% low reading: "
          f"bar {1e3*0.01/sv_bar:.1f} um, disc {1e3*0.01/sv_disc:.1f} um")
    sv = np.where(d.campaign == "seed", sv_bar, sv_disc)
    for delta_um in (0, 5, 10):
        dd = d.copy()
        dd["rho"] = d.rho * (1 + 1e-3 * delta_um * sv)
        m = g.fit_all(dd)
        pf = g.acquisition(m, dd, Xc)[1]
        print(f"delta = {delta_um:2d} um: mean density {dd.rho.mean():.3f}, largest Pr_feas {pf.max():.2f}")
    print()


def latent_var_reduction(d, x, Xhull):
    """Mean latent density variance over the seed hull before and after one more build at x.

    The latent variance does not depend on the measured value once the hyperparameters are
    fixed, so the fitted hyperparameters are kept and the build is added at its predicted mean.
    """
    m = g.fit_all(d)["rho"]
    v0 = m.predict(Xhull)[1] ** 2
    X = np.vstack([g.features(d.P, d.v, d.h), x])
    y = np.append(d.rho.values, m.predict(x)[0])
    s = np.append(g.noise(d, "rho"), g.S_RHO)
    m1 = g.PhysicsGP(g.PRIORS["rho"])
    m1.mu_x, m1.sd_x = m.mu_x, m.sd_x
    m1.Z, m1.r, m1.s = (X - m.mu_x) / m.sd_x, y - m.prior(X), s
    m1.sf2, m1.ell, m1.beta = m.sf2, m.ell, (y - m.prior(X)).mean()
    m1.Ki = np.linalg.inv(g.matern52(m1.Z, m1.Z, m1.sf2, m1.ell) + np.diag(s ** 2))
    one = np.ones(len(X))
    m1.beta_var_den = one @ m1.Ki @ one
    m1.alpha = m1.Ki @ (m1.r - m1.beta)
    v1 = m1.predict(Xhull)[1] ** 2
    return 1 - v1.mean() / v0.mean()


def family_separation(d, X):
    """Mean density prediction of the 80-90 minus the 100-110 J/mm3 prior families, in units of s_rho."""
    fam = {}
    for name, centres in (("low", (80, 85, 90)), ("high", (100, 105, 110))):
        preds = []
        for c in centres:
            m = g.PhysicsGP(lambda Z, c=c: g.prior_rho(Z, ved_opt=c)).fit(
                g.features(d.P, d.v, d.h), d.rho.values, g.noise(d, "rho"))
            preds.append(m.predict(X)[0])
        fam[name] = np.mean(preds, axis=0)
    return (fam["low"] - fam["high"]) / g.S_RHO


def simulate(d, x, Xc, n=N_SIM, seed=SIM_SEED):
    """Largest Pr_feas over the grid after one simulated build at x (density and hardness drawn)."""
    rng = np.random.default_rng(seed)
    m = g.fit_all(d)
    mr, sr = m["rho"].predict(x)
    mh, sh = m["HV"].predict(x)
    sh_new = g.S_HV1 / np.sqrt(N_INDENTS_NEW)
    out = []
    for _ in range(n):
        new = pd.DataFrame({"build": ["V"], "campaign": ["virtual"], "role": ["virtual"],
                            "P": [x[0, 0]], "v": [x[0, 1]], "h": [x[0, 2]],
                            "rho": [rng.normal(mr[0], np.hypot(sr[0], g.S_RHO))],
                            "HV": [rng.normal(mh[0], np.hypot(sh[0], sh_new))],
                            "HV_sd": [np.nan], "n_indents": [N_INDENTS_NEW], "p2D": [np.nan]})
        dd = pd.concat([d, new], ignore_index=True)
        X = g.features(dd.P, dd.v, dd.h)
        models = {t: g.PhysicsGP(g.PRIORS[t]).fit(X, g.target_values(dd, t), g.noise(dd, t))
                  for t in ("rho", "HV")}
        out.append(g.acquisition(models, dd, Xc)[1].max())
    return np.array(out)


def table_a4(d, seed, Xc):
    m = g.fit_all(d)
    a, pf, mr, mh, _ = g.acquisition(m, d, Xc)
    hull = g.in_hull(seed, Xc[:, 0], Xc[:, 1], Xc[:, 2])
    sep = family_separation(d, Xc)
    ok = (Xc[:, 0] <= P_LIMIT) & (pf >= 0.2)
    j = np.where(ok)[0][np.argmax(np.abs(sep[ok]))]
    points = {"Replicate of A": g.features([265], [1100], [0.071]),
              "Acquisition maximum": Xc[[a.argmax()]],
              "Prior discrimination": Xc[[j]]}
    print("== Table A4: virtual twelfth build with the eleven-build surrogate")
    print(f"largest Pr_feas now {pf.max():.2f}; {N_SIM} simulated outcomes (seed {SIM_SEED}), "
          f"density and hardness drawn, {N_INDENTS_NEW} indents")
    print(f"{'objective':22s}{'P/v/h':>18s}{'VED':>6s}{'rho (95% PI)':>22s}{'HV':>5s}{'Pr':>6s}"
          f"{'d var':>7s}{'Sep':>6s}{'E[max Pr]':>10s}{'P(>0.5)':>8s}")
    for name, x in points.items():
        r, sr = m["rho"].predict(x)
        hv = m["HV"].predict(x)[0][0]
        pr = g.acquisition(m, d, x)[1][0]
        pi = 1.96 * np.hypot(sr[0], g.S_RHO)
        sim = simulate(d, x, Xc)
        print(f"{name:22s}{f'{x[0,0]:.0f}/{x[0,1]:.0f}/{x[0,2]:.3f}':>18s}{x[0,3]:6.1f}"
              f"{f'{r[0]:.3f} ({r[0]-pi:.2f}-{r[0]+pi:.2f})':>22s}{hv:5.0f}{pr:6.2f}"
              f"{100*latent_var_reduction(d, x, Xc[hull]):6.1f}%{family_separation(d, x)[0]:6.2f}"
              f"{sim.mean():10.2f}{np.mean(sim > 0.5):8.2f}")
    with Settings(0.04):
        sim = simulate(d, points["Acquisition maximum"], Xc)
        now = g.acquisition(g.fit_all(d), d, Xc)[1].max()
    print(f"density noise halved (0.04): largest Pr_feas now {now:.2f}; after a build at the "
          f"acquisition maximum E[max Pr] {sim.mean():.2f}, P(>0.5) {np.mean(sim > 0.5):.2f}")


if __name__ == "__main__":
    d = pd.read_csv(_builds_csv())
    seed = d[d.campaign == "seed"].reset_index(drop=True)
    follow = d[d.campaign != "seed"].reset_index(drop=True)
    Xc = g.features(*g.candidates())
    table_a3(seed, follow, Xc)
    air_layer_scenario(d, Xc)
    table_a4(d, seed, Xc)
