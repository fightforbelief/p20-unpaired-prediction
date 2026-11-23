"""Sharp scalar conditional prediction for a Gaussian covariance completion."""
import heapq
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri


@dataclass
class Radius:
    lower: float
    upper: float
    coupling: np.ndarray


def folded_quantile(mean, sd, alpha):
    """Quantile of the absolute value of a normal random variable."""
    mean = abs(mean)
    if sd == 0:
        return mean
    z = -ndtri(alpha / 2)
    # Solve for the offset from mean, retaining precision at large mean/sd.
    ratio = mean / sd
    offset = brentq(lambda t: ndtr(-t) + ndtr(-2 * ratio - t) - alpha,
                    0, z + 1, xtol=1e-14)
    return mean + sd * offset


def intercept(slope, alpha):
    if slope == 0:
        return -ndtri(alpha / 2)
    if slope == 1:
        return -ndtri(alpha)
    target = np.arctanh(slope)
    b = brentq(lambda x: x * folded_quantile(x, 1., alpha) - target,
               0, math.sqrt(target) + 1)
    return folded_quantile(b, 1., alpha) - b * slope


def support(d, g, a, k):
    """Dual upper bound and a feasible primal vector for a tangent plane."""
    # Zero gains are removed by the caller.
    gains2 = d * d
    weights = (a * g) ** 2

    def derivative(lam):
        denom = gains2 + lam * (1 - gains2)
        return (k / (1 - lam)) ** 2 - np.sum(weights * (1 - gains2) / denom**2)

    if derivative(0.) >= 0:
        lam = 0.
    else:
        lam = brentq(derivative, 0., np.nextafter(1., 0.), xtol=5e-15)
    denom = gains2 + lam * (1 - gains2)
    value = np.sqrt(np.sum(weights / denom) + k * k / (1 - lam))
    u = a * g / (denom * value)
    return value, u / max(1., np.linalg.norm(u))


def sharp_radius(gains, observation, alpha=0.05, tol=0.0005):
    """Bracket the shortest valid half-length, in target standard deviations.

    gains are canonical correlations in [0,1]. observation is in whitened
    left singular coordinates. The coupling returned attains the lower bound.
    """
    d = np.asarray(gains, dtype=float)
    w = np.asarray(observation, dtype=float)
    if (d.ndim != 1 or w.shape != d.shape or not np.isfinite(w).all()
            or not np.isfinite(d).all() or np.any((d < 0) | (d > 1))
            or not 0 < alpha < .5 or not np.isfinite(tol) or tol <= 0):
        raise ValueError("invalid spectrum, observation, level or tolerance")
    active = (d > 0) & (w != 0)
    g = (d * w)[active]
    d = d[active]
    best = -ndtri(alpha / 2)
    witness = np.zeros_like(w)
    if not len(g):
        return Radius(best, best, witness)

    def probe(a, k):
        nonlocal best
        _, u = support(d, g, a, k)
        sd = np.sqrt(max(0., 1 - np.sum((d * u)**2)))
        h = folded_quantile(g @ u, sd, alpha)
        if h > best:
            best = h
            witness[active] = u

    queue = []

    def enqueue(left, right, k_left, k_right):
        upper, _ = support(d, g, right, k_left)
        heapq.heappush(queue, (-upper, left, right, k_left, k_right))

    k0, k1 = intercept(0, alpha), intercept(1, alpha)
    probe(1, k1)
    enqueue(0, 1, k0, k1)
    while -queue[0][0] > best + tol:
        _, left, right, kl, kr = heapq.heappop(queue)
        mid = (left + right) / 2
        if mid == left or mid == right:
            raise ArithmeticError("requested tolerance exceeds floating-point resolution")
        km = intercept(mid, alpha)
        probe(mid, km)
        enqueue(left, mid, kl, km)
        enqueue(mid, right, km, kr)
    return Radius(best, max(best, -queue[0][0]), witness)
