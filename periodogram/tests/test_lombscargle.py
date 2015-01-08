import numpy as np
from numpy.testing import assert_allclose, assert_array_less, assert_equal

from ..lomb_scargle import LombScargle

def make_sine(N=100, err=0.05, rseed=None):
    rng = np.random.RandomState(rseed)
    t = 2 * np.pi * rng.rand(N)
    y = np.sin(t) + err * rng.randn(N)
    return t, y, err


def test_sine():
    t, y, dy = make_sine(N=100, err=0.05, rseed=0)

    model = LombScargle(pmin=0.01, pmax=4*np.pi, linspace=True)
    model.fit(t, y, dy)
    pbest = model.period_search()

    assert_allclose(pbest, 2*np.pi, 0.01)

    

