"""
Initial surrogate of the paper (Sect. 4.7.1).

This model selected builds A and C. It uses six inputs, learns a single noise
variance by maximum likelihood (WhiteKernel) and is fitted to the seven seed
builds only. The model code is unchanged from the seven-build implementation kept
in the repository history (commit 50172b8); only the data loading and the
follow-up test at the end were added.

Reference:
    Zeybek et al., "A closed-loop active learning framework for laser powder bed fusion calibration under data scarcity", 
    Journal of Intelligent Manufacturing (under review).

"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

# Fixed layer thickness across the campaign (mm)
LAYER_THICKNESS = 0.04

# Feature order used everywhere: [P, v, h, VED, cooling-rate index, thermal-input index]
FEATURES = ["P", "v", "h", "VED", "Cdot", "TI"]


def load_data(csv_path):
    """Load the seven-build seed dataset and compute physics-derived features."""
    raw = pd.read_csv(csv_path)
    df = pd.DataFrame({
        "S": raw["sample"],
        "P": raw["P_W"].astype(float),
        "v": raw["v_mm_s"].astype(float),
        "h": raw["h_mm"].astype(float),
        "rho": raw["density_g_cm3"].astype(float),
        "HV": raw["HV1_mean"].astype(float),
        "HV_sd": raw["HV1_std"].astype(float),
        "p2D": raw["p2D_pct"].astype(float),
    })
    t = LAYER_THICKNESS
    df["VED"] = df.P / (df.v * df.h * t)            # volumetric energy density
    df["Cdot"] = 1000.0 * np.sqrt(df.v) / df.P      # cooling-rate index
    df["TI"] = df.P / np.sqrt(df.v)                 # thermal-input index
    df["logp"] = np.log10(df.p2D)                   # log10 porosity (regression target)
    X = df[FEATURES].values
    return df, X


# ---------------------------------------------------------------------------
# Physics-informed prior mean functions (parameters fixed a priori, not fitted)
# ---------------------------------------------------------------------------

def prior_density(Xf):
    """Gaussian peak of as-built density vs VED, centred on the alloy optimum."""
    VED = Xf[:, 3]
    rho_min, rho_max, VED_opt, sigma = 8.20, 8.44, 85.0, 18.0
    return rho_min + (rho_max - rho_min) * np.exp(-(VED - VED_opt) ** 2 / (2 * sigma ** 2))


def prior_hardness(Xf):
    """Hall-Petch-flavoured hardness increasing with cooling-rate index."""
    Cdot = Xf[:, 4]
    return 265.0 + 20.0 * (Cdot / 115.0) ** 0.25


def prior_logp(Xf):
    """Parabolic-in-VED log-porosity prior (rises away from the optimum)."""
    VED = Xf[:, 3]
    return np.clip(-0.8 + 8e-4 * (VED - 85.0) ** 2, -1.5, 0.5)


# ---------------------------------------------------------------------------
# Kernel and GP estimators
# ---------------------------------------------------------------------------

def make_kernel():
    """Matern-5/2 with bounded hyperparameters and a white-noise floor."""
    return (ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(1.0, (1e-2, 1e3), nu=2.5)
            + WhiteKernel(1e-3, (1e-6, 1e-1)))


def make_gpr():
    """Standard GPR with five restarts and target normalisation."""
    return GaussianProcessRegressor(
        kernel=make_kernel(), normalize_y=True,
        n_restarts_optimizer=5, random_state=0)


class Hybrid:
    """Hybrid-GPR: physics prior mean + zero-mean residual GP (Kennedy-O'Hagan)."""

    def __init__(self, prior_fn):
        self.prior_fn = prior_fn

    def fit(self, Xr, y):
        self.scaler = StandardScaler().fit(Xr)
        self.gp = make_gpr().fit(self.scaler.transform(Xr), y - self.prior_fn(Xr))
        return self

    def predict(self, Xq, return_std=False):
        Xq = np.atleast_2d(Xq)
        mu, sd = self.gp.predict(self.scaler.transform(Xq), return_std=True)
        mu = mu + self.prior_fn(Xq)
        return (mu, sd) if return_std else mu


# ---------------------------------------------------------------------------
# Decision layer: Cost-Aware CEI and Pareto extraction
# ---------------------------------------------------------------------------

def cost_aware_cei(mu, sd, f_star, feasible, P, lam=0.005, centreline=275.0):
    """Cost-Aware Constrained Expected Improvement acquisition score."""
    z = (mu - f_star) / np.maximum(sd, 1e-9)
    ei = (mu - f_star) * norm.cdf(z) + sd * norm.pdf(z)
    cost = 1.0 + lam * np.abs(P - centreline)
    return np.where(feasible, ei / cost, -np.inf)


def pareto_front(objectives):
    """Boolean mask of non-dominated rows (all objectives maximised)."""
    O = np.asarray(objectives, dtype=float)
    n = len(O)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        keep[i] = not np.any(np.all(O >= O[i], axis=1) & np.any(O > O[i], axis=1))
    return keep


def bootstrap_ci(errors, n_boot=1000, seed=0):
    """95% bootstrap confidence interval on a mean error."""
    rng = np.random.default_rng(seed)
    means = [rng.choice(errors, len(errors), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _builds_csv():
    """Locate builds.csv next to the script or in ../data."""
    from pathlib import Path
    here = Path(__file__).resolve().parent
    for p in (here / "builds.csv", here.parent / "data" / "builds.csv", Path.cwd() / "builds.csv"):
        if p.exists():
            return p
    raise FileNotFoundError("builds.csv not found next to the script or in ../data")


def _frame(d):
    """Convert rows of builds.csv to the input format of load_data."""
    return pd.DataFrame({"sample": d.build.values, "P_W": d.P.values, "v_mm_s": d.v.values,
                         "h_mm": d.h.values, "density_g_cm3": d.rho.values,
                         "HV1_mean": d.HV.values, "HV1_std": d.HV_sd.values, "p2D_pct": d.p2D.values})


if __name__ == "__main__":
    import io
    import warnings
    warnings.simplefilter("ignore")
    b = pd.read_csv(_builds_csv())
    load = lambda d: load_data(io.StringIO(_frame(d).to_csv(index=False)))
    df, X = load(b[b.campaign == "seed"])
    dq, Xq = load(b[b.campaign != "seed"])
    m_rho = Hybrid(prior_density).fit(X, df.rho.values)
    m_hv = Hybrid(prior_hardness).fit(X, df.HV.values)
    m_lp = Hybrid(prior_logp).fit(X, df.logp.values)
    r, rs = m_rho.predict(Xq, return_std=True)
    hv, hs = m_hv.predict(Xq, return_std=True)
    lp, ls = m_lp.predict(Xq, return_std=True)
    p_joint = (1 - norm.cdf((8.40 - r) / rs)) * (1 - norm.cdf((275.0 - hv) / hs))
    print("Initial surrogate fitted to the seven seed builds; prediction at the follow-up builds")
    print(f"{'build':>5} {'rho_pred':>9} {'sd':>6} {'rho_meas':>9} {'|err|':>6} {'z':>6} "
          f"{'HV_pred':>8} {'sd':>5} {'logp_pred':>10} {'sd':>6} {'logp_meas':>10} {'P(rho>=8.40, HV>=275)':>22}")
    for i, s in enumerate(dq.S):
        z = (dq.rho.values[i] - r[i]) / rs[i]
        print(f"{s:>5} {r[i]:9.3f} {rs[i]:6.3f} {dq.rho.values[i]:9.3f} {abs(r[i] - dq.rho.values[i]):6.4f} {z:6.2f} "
              f"{hv[i]:8.1f} {hs[i]:5.1f} {lp[i]:10.2f} {ls[i]:6.3f} {dq.logp.values[i]:10.2f} {p_joint[i]:22.2f}")
