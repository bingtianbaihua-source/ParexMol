import unittest

import numpy as np
from scipy.optimize import check_grad

from utils.frontier_gp import components,likelihood,fitted_posterior
from utils.frontier_surrogate import features


class FittedGPTests(unittest.TestCase):
    def setUp(self):
        self.x=np.stack([features(s) for s in
            ['CCCCCC','CCCCCO','CCCCCN','c1ccccc1','Cc1ccccc1','Oc1ccccc1']])

    def test_likelihood_gradient_matches_numerical_differentiation(self):
        t,d=components(self.x,self.x);target=np.random.default_rng(4).normal(size=len(self.x))
        for p in [[.7,0.,np.log(.05)],[.2,-.6,np.log(.2)],[.9,.8,np.log(.03)]]:
            error=check_grad(lambda z:likelihood(z,t,d,target)[0],
                             lambda z:likelihood(z,t,d,target)[1],np.asarray(p))
            self.assertLess(error,2e-5)

    def test_objectives_are_fitted_independently_and_dimensions_are_not_fixed(self):
        rng=np.random.default_rng(8)
        for dimension in [2,5,9]:
            y=rng.uniform(.1,.9,size=(len(self.x),dimension))
            mu,std,fits=fitted_posterior(self.x,y,self.x[:2])
            self.assertEqual(mu.shape,(2,dimension));self.assertEqual(std.shape,mu.shape)
            self.assertTrue(np.isfinite(mu).all());self.assertTrue((std>0).all())
            self.assertTrue(all(f['fitted_nll']<=f['initial_nll'] for f in fits))
            changed=y.copy();changed[:,-1]=rng.uniform(.1,.9,size=len(y))
            other_mu,other_std,other_fits=fitted_posterior(self.x,changed,self.x[:2])
            np.testing.assert_allclose(mu[:,:-1],other_mu[:,:-1],atol=1e-12,rtol=0)
            np.testing.assert_allclose(std[:,:-1],other_std[:,:-1],atol=1e-12,rtol=0)
            self.assertEqual(fits[:-1],other_fits[:-1])


if __name__=='__main__':unittest.main()
