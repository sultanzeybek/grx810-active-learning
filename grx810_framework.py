"""
Closed-loop active learning framework for sparse-data L-PBF calibration of GRX-810.

Core library: physics-informed Hybrid Gaussian Process Regression (Hybrid-GPR),
literature-anchored prior mean functions, Constrained Expected Improvement (CEI),
and Pareto-front extraction.

Reference:
    Zeybek et al., "A closed-loop active learning framework for laser powder bed fusion calibration under data scarcity", 
    Journal of Intelligent Manufacturing (under review).

Usage:
    from grx810_framework import load_data, Hybrid, prior_density, prior_hardness, prior_logp
    df, X = load_data("data/builds.csv")
    model = Hybrid(prior_density).fit(X, df.density.values)
    mu, sd = model.predict(X, return_std=True)
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


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df, X = load_data(os.path.join(here, "data", "builds.csv"))
    print(f"Loaded {len(df)} builds. VED range: {df.VED.min():.1f}-{df.VED.max():.1f} J/mm^3")
    m = Hybrid(prior_density).fit(X, df.rho.values)
    mu, sd = m.predict(X, return_std=True)
    print("Hybrid-GPR density posterior at training points:",
          np.array2string(mu, precision=3))
