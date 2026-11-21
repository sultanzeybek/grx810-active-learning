"""Hybrid-GPR: physics-informed Gaussian process with fixed measurement noise and cost-aware CEI.

Implements the model of Sects. 2.1-2.3 of the paper: five inputs, physics prior means,
constant offset, Matern-5/2 residual GP, fixed noise scales, probability of feasibility
and the cost-aware constrained expected improvement.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.stats import norm

T_LAYER = 0.04
S_RHO, S_P, S_HV1 = 0.08, 0.27, 8.8
RHO_THR, HV_THR = 8.40, 275.0


def features(P, v, h):
    P, v, h = map(np.asarray, (P, v, h))
    ved = P / (v * h * T_LAYER)
    cdot = 1000 * np.sqrt(v) / P
    return np.column_stack([P, v, h, ved, cdot])


# ---- prior means (Table 2) -------------------------------------------------
def prior_rho(X, ved_opt=85.0, rho_max=8.44, rho_min=8.20, sig=18.0):
    return rho_min + (rho_max - rho_min) * np.exp(-(X[:, 3] - ved_opt) ** 2 / (2 * sig ** 2))


def prior_hv(X, H0=265.0, dH=20.0, cref=115.0):
    return H0 + dH * (X[:, 4] / cref) ** 0.25


def prior_logp(X, a0=-0.8, a2=8e-4):
    return np.clip(a0 + a2 * (X[:, 3] - 85.0) ** 2, -1.5, 0.5)


def zero(X):
    return np.zeros(len(X))


# ---- GP ----------------------------------------------------------------------
def matern52(A, B, sf2, ell):
    r = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    a = np.sqrt(5) * r / ell
    return sf2 * (1 + a + a ** 2 / 3) * np.exp(-a)


class PhysicsGP:
    """y = m(x) + beta + delta(x) + eps, eps ~ N(0, s_i^2) fixed; beta by mean residual."""

    def __init__(self, prior, restarts=5, ell_min=0.5, seed=0):
        self.prior, self.restarts, self.ell_min, self.seed = prior, restarts, ell_min, seed

    def fit(self, X, y, s):
        self.mu_x, self.sd_x = X.mean(0), X.std(0)
        self.sd_x[self.sd_x == 0] = 1
        Z = (X - self.mu_x) / self.sd_x
        r = y - self.prior(X)
        self.beta = r.mean()
        rc = r - self.beta
        scale = max(rc.var(), 1e-12)  # sigma_f^2 bounds are given in normalised units
        S2 = np.diag(s ** 2)

        def nll(th):
            sf2, ell = np.exp(th)
            K = matern52(Z, Z, sf2, ell) + S2 + 1e-12 * np.eye(len(Z))
            L = np.linalg.cholesky(K)
            a = np.linalg.solve(L.T, np.linalg.solve(L, rc))
            return 0.5 * rc @ a + np.log(np.diag(L)).sum()

        bnds = [(np.log(1e-3 * scale), np.log(1e3 * scale)), (np.log(self.ell_min), np.log(1e3))]
        rng = np.random.default_rng(self.seed)
        starts = [np.log([scale, 1.0])] + [
            [rng.uniform(*bnds[0]), rng.uniform(*bnds[1])] for _ in range(self.restarts)]
        best = min((minimize(nll, x0, bounds=bnds, method="L-BFGS-B") for x0 in starts),
                   key=lambda o: o.fun)
        self.sf2, self.ell = np.exp(best.x)
        self.Z, self.r, self.s = Z, r, s
        Ky = matern52(Z, Z, self.sf2, self.ell) + S2
        self.Ki = np.linalg.inv(Ky)
        one = np.ones(len(Z))
        self.beta_var_den = one @ self.Ki @ one
        self.alpha = self.Ki @ (r - self.beta)
        return self

    def predict(self, X):
        Zq = (X - self.mu_x) / self.sd_x
        ks = matern52(self.Z, Zq, self.sf2, self.ell)
        mean = self.prior(X) + self.beta + ks.T @ self.alpha
        one = np.ones(len(self.Z))
        u = 1 - one @ self.Ki @ ks
        var = self.sf2 - np.einsum("ij,ik,kj->j", ks, self.Ki, ks) + u ** 2 / self.beta_var_den
        return mean, np.sqrt(np.maximum(var, 1e-12))


def noise(df, target):
    if target == "rho":
        return np.full(len(df), S_RHO)
    if target == "logp":
        return np.full(len(df), S_P)
    return S_HV1 / np.sqrt(df.n_indents.values)


def target_values(df, target):
    return np.log10(df.p2D.values) if target == "logp" else df[target].values


PRIORS = {"rho": prior_rho, "HV": prior_hv, "logp": prior_logp}


def fit_all(df, physics=True):
    X = features(df.P, df.v, df.h)
    return {t: PhysicsGP(PRIORS[t] if physics else zero).fit(X, target_values(df, t), noise(df, t))
            for t in ("rho", "HV", "logp")}


def candidates(n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.uniform(150, 400, n); v = rng.uniform(600, 1400, n); h = rng.uniform(0.06, 0.10, n)
    return P, v, h


def in_hull(df, P, v, h):
    pts = np.column_stack([df.P, df.v, df.h])
    mu, sd = pts.mean(0), pts.std(0)
    tri = Delaunay((pts - mu) / sd)
    return tri.find_simplex((np.column_stack([P, v, h]) - mu) / sd) >= 0


def acquisition(models, df, Xc, lam=0.005):
    X = features(df.P, df.v, df.h)
    mr, sr = models["rho"].predict(Xc)
    mh, sh = models["HV"].predict(Xc)
    fstar = models["rho"].predict(X)[0].max()
    a = max(fstar, RHO_THR)
    ei = (mr - fstar) * norm.cdf((mr - a) / sr) + sr * norm.pdf((a - mr) / sr)
    cost = 1 + lam * np.abs(Xc[:, 0] - 275)
    pfeas = norm.cdf((mr - RHO_THR) / sr) * norm.cdf((mh - HV_THR) / sh)
    return ei * norm.cdf((mh - HV_THR) / sh) / cost, pfeas, mr, mh, fstar
