"""Regression checks for the fitted objective, ties, direction, and boundaries."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Algorithm.apply_map import apply_map
from Algorithm.comet_numpy import Objective, fit_projected
from Algorithm.mos_numpy import mos

class Algorithms(unittest.TestCase):
    def test_endpoint_and_interior_mapping(self):
        np.testing.assert_allclose(apply_map([1,2,2,4], [3,5,5,9], [-np.inf,1,1.5,3,4,np.inf]), [3,3,4,7,9,9])
        np.testing.assert_array_equal(apply_map([2,2], [7,7], [-100,2,100]), [7,7,7])
        with self.assertRaises(AssertionError):
            apply_map([1,1], [2,3], [1])

    def test_analytic_gradient(self):
        x=np.linspace(.1,2,24);y=np.sqrt(x);o=Objective(x,y,.5,1)
        a=o.initial;v=np.random.default_rng(12).normal(size=len(a));v/=np.linalg.norm(v)
        numeric=(o.fun(a+1e-5*v)[0]-o.fun(a-1e-5*v)[0])/2e-5
        self.assertLess(abs(numeric-o.fun(a)[1]@v),1e-7)

    def test_projected_fit_and_equal_inputs(self):
        x=np.repeat(np.linspace(.1,2,20),2);y=np.sqrt(x)
        for response in [y,-y]:
            z,fits=fit_projected(x,response,.5)
            self.assertTrue(all(f['success'] for f in fits))
            u,first,inverse=np.unique(x,return_index=True,return_inverse=True)
            self.assertTrue(np.all(np.diff(z[first])>0))
            np.testing.assert_array_equal(z,z[first][inverse])
            best=min(fits,key=lambda f:f['loss'])
            loss=-abs(np.corrcoef(z,response)[0,1])+.5*np.sum((z-x)**2)/np.sum((x-x.mean())**2)
            self.assertAlmostEqual(loss,best['loss'],places=11)
            np.testing.assert_allclose(apply_map(x,z,[-1e50,1e50]),[z.min(),z.max()])

    def test_mos_plateau_and_constant_scores(self):
        x=np.arange(8,dtype=float);y=np.array([0,1,1,0,1,2,2,1.])
        z=mos(x,y)
        self.assertTrue(np.all(np.diff(z)>=0))
        self.assertLess(len(np.unique(z)),len(np.unique(x)))
        with np.errstate(invalid='ignore'):
            np.testing.assert_array_equal(mos(x,np.ones(8)),np.full(8,x.mean()))

if __name__=='__main__':
    unittest.main()
