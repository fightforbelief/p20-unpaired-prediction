"""Finite paired-archive prediction in whitened latent coordinates."""
from functools import lru_cache
import heapq

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, ndtri


@lru_cache(maxsize=None)
def marginal_drop(n, reliability=1.):
    """E[(1+c² chi²_n)^-1] minus its value after one additional score."""
    def integrand(t):
        a = np.log1p(2 * reliability**2 * t) / 2
        return np.exp(-t - n*a) * (-np.expm1(-a))
    return quad(integrand, 0, np.inf, epsabs=2e-13)[0]


def allocate(gains, budget, policy="greedy", order=None, reliability=None):
    d = np.asarray(gains, dtype=float)
    if d.ndim != 1 or not len(d) or not np.isfinite(d).all() or np.any((d<0)|(d>1)):
        raise ValueError("gains must be a nonempty vector in [0,1]")
    if not isinstance(budget, (int, np.integer)) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    c = np.ones_like(d) if reliability is None else np.asarray(reliability, dtype=float)
    if c.shape != d.shape or not np.isfinite(c).all() or np.any((c<0)|(c>1)):
        raise ValueError("invalid score reliabilities")
    if policy in {"greedy", "mean_info"}:
        def benefit(i,n):
            if policy == "greedy":
                drop = marginal_drop(n,float(c[i]))
            else:
                c2 = c[i]**2
                drop = c2/((1+n*c2)*(1+(n+1)*c2))
            return d[i]**2 * drop
        heap = [(-benefit(i,0),i) for i in range(len(d))]
        heapq.heapify(heap)
        counts = np.zeros(len(d), dtype=int)
        sequence = []
        for _ in range(budget):
            _, i = heapq.heappop(heap)
            counts[i] += 1
            sequence.append(i)
            heapq.heappush(heap, (-benefit(i,int(counts[i])),i))
        return np.asarray(sequence, dtype=int)
    if policy == "top4":
        return np.resize(np.argsort(-d,kind="stable")[:4],budget)
    if policy == "balanced":
        return np.resize(np.arange(len(d)) if order is None else order,budget)
    if policy == "proportional":
        shares = budget * (d/d.sum() if d.sum() else np.full(len(d),1/len(d)))
        counts = shares.astype(int)
        counts[np.argsort(counts-shares,kind="stable")[:budget-counts.sum()]] += 1
        return np.repeat(np.arange(len(d)), counts)
    raise ValueError(f"unknown allocation {policy}")


def folded_radius(bias, sd, alpha=.05):
    """Vectorized folded-normal quantile. Return the feasible bisection endpoint."""
    bias, sd = np.broadcast_arrays(bias, sd)
    if not 0 < alpha < .5 or not np.isfinite(bias).all() or not np.isfinite(sd).all() or np.any(bias<0) or np.any(sd<=0):
        raise ValueError("require finite nonnegative bias and positive scale")
    ratio = bias / sd
    lo = np.zeros_like(ratio)
    hi = np.full(ratio.shape,-ndtri(alpha/2),dtype=float)
    for _ in range(42):
        mid = (hi+lo)/2
        smaller = ndtr(-mid)+ndtr(-2*ratio-mid) <= alpha
        hi = np.where(smaller,mid,hi)
        lo = np.where(smaller,lo,mid)
    return bias+sd*hi


def predict_scores(rows, targets, g):
    """rows are conditional means E[X_j | calibration score j]."""
    rows = np.asarray(rows, dtype=float)
    g = np.atleast_2d(g)
    residual = np.linalg.solve(np.eye(rows.shape[1])+rows.T@rows,g.T).T
    weights = residual @ rows.T
    energy = np.sum(weights**2,axis=1)
    bias2 = np.sum(residual**2,axis=1)
    return weights @ targets, folded_radius(np.sqrt(bias2),np.sqrt(1+energy)), bias2+energy


def predict_axes(indices, scores, targets, g):
    """Diagonal specialization. scores already include any reliability factor."""
    p = g.shape[1]
    energy = np.bincount(indices, weights=scores**2, minlength=p)
    sums = np.bincount(indices, weights=scores*targets, minlength=p)
    residual = g / (1+energy)
    variance = (residual**2) @ energy
    bias2 = np.sum(residual**2,axis=1)
    return residual @ sums, folded_radius(np.sqrt(bias2),np.sqrt(1+variance)), bias2+variance
