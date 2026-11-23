import unittest
import numpy as np
from scipy.special import ndtr, ndtri

from intervals import folded_quantile, sharp_radius


class RadiusTests(unittest.TestCase):
    def test_null_and_noiseless(self):
        z = -ndtri(.025)
        self.assertAlmostEqual(sharp_radius([0], [5]).upper, z)
        for alpha in [.01, .05, .2]:
            for r in [0, .8, 1.2, 3, 10]:
                result = sharp_radius([1], [r], alpha, .0002)
                grid = np.linspace(0, np.pi / 2, 2001)
                ref = max(folded_quantile(r*np.sin(t), np.cos(t), alpha) for t in grid)
                self.assertLessEqual(ref, result.upper + 1e-10)
                self.assertLess(result.upper - ref, .00021)
                self.assertGreaterEqual(result.lower + 1e-10, -ndtri(alpha / 2))

    def test_anisotropic_directions(self):
        d = np.array([.04, .97])
        w = np.array([-5., 2.5])
        result = sharp_radius(d, w, tol=.0003)
        u = result.coupling
        self.assertLessEqual(np.linalg.norm(u), 1 + 1e-12)
        mean, sd = (d*w) @ u, np.sqrt(1 - np.sum((d*u)**2))
        self.assertAlmostEqual(ndtr((result.lower-mean)/sd) -
                               ndtr((-result.lower-mean)/sd), .95)
        rng = np.random.default_rng(123)
        angles = rng.uniform(0, 2*np.pi, 20000)
        radii = np.r_[np.ones(10000), rng.uniform(size=10000)]
        v = radii[:,None] * np.column_stack((np.cos(angles), np.sin(angles)))
        m, s = (v*d) @ w, np.sqrt(1 - np.sum((v*d)**2, axis=1))
        coverage = ndtr((result.upper-m)/s) - ndtr((-result.upper-m)/s)
        self.assertGreaterEqual(coverage.min(), .95 - 1e-12)
        self.assertLess(coverage.min(), .951)

    def test_equal_gains_rotation(self):
        a = sharp_radius([.6,.6,0], [3,4,17], tol=1e-4)
        b = sharp_radius([.6], [5], tol=1e-4)
        self.assertAlmostEqual(a.lower,b.lower)
        self.assertAlmostEqual(a.upper,b.upper)

    def test_invalid(self):
        for d,w in [([1.1],[1]), ([.5],[np.nan]), ([0,1],[1])]:
            with self.assertRaises(ValueError):
                sharp_radius(d,w)
