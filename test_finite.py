import itertools
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from finite_design import allocate, folded_radius, predict_axes, predict_scores


class FiniteTests(unittest.TestCase):
    def test_greedy_exhaustive_with_measurement_error(self):
        d=np.array([.95,.7,.4])
        c=np.array([.2,1.,.6])
        def cost(counts):
            return sum(x*x*quad(lambda t: np.exp(-t)*(1+2*r*r*t)**(-n/2),
                                0,np.inf)[0] for x,r,n in zip(d,c,counts))
        got=np.bincount(allocate(d,6,reliability=c),minlength=3)
        values=[cost(v) for v in itertools.product(range(7),repeat=3) if sum(v)==6]
        self.assertAlmostEqual(cost(got),min(values))

    def test_dense_diagonal_agree(self):
        rng=np.random.default_rng(5)
        idx=np.array([0,1,1,3]); scores=rng.normal(size=4)
        raw=np.eye(5)[idx]*scores[:,None]
        g=rng.normal(size=(6,5)); y=rng.normal(size=4)
        for a,b in zip(predict_scores(raw,y,g), predict_axes(idx,scores,y,g)):
            np.testing.assert_allclose(a,b,atol=1e-12)

    def test_center_radius_and_modulus(self):
        g=np.array([[1.,2.]])
        a=np.array([[.5,1.]])
        pred,half,m2=predict_scores(a,[1.],g)
        self.assertAlmostEqual(m2.item(),(g@np.linalg.inv(np.eye(2)+a.T@a)@g.T).item())
        weight=pred.item()
        for theta in np.linspace(0,2*np.pi,501):
            u=np.array([np.cos(theta),np.sin(theta)])
            bias=((g-weight*a)@u).item()
            var=1-np.sum((.5*u)**2)+weight**2*(1-(np.array([1,2])@u/np.sqrt(5))**2)
            self.assertGreaterEqual((ndtr((half-bias)/np.sqrt(var))-ndtr((-half-bias)/np.sqrt(var))).item(),.95-1e-12)
        self.assertAlmostEqual(folded_radius(0.,1.).item(),1.95996398454)
