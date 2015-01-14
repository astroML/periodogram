import numpy as np
from numpy.testing import assert_allclose, assert_array_less, assert_equal

import logging

import matplotlib.pyplot as plt

from ..acf import ACF

def make_sine(N=1000, P=100, err=0.05, rseed=None, even=True):
    rng = np.random.RandomState(rseed)

    if even:
        t = np.arange(N) #evenly-sampled data
    else:
        t = np.sort(rng.rand(N)*N)
    y = np.ones(len(t))
    
    y *= np.sin(2*np.pi*t/P)
    y += rng.randn(N)*err
    
    return t,y

def test_single_sine_standard(P=100):
    t,y = make_sine(P=P)
    a = ACF()
    a.fit(t,y)
    pbest = a.period_search()
    assert_allclose(pbest, P, 0.01)

## TODO
#def test_single_sine_scargle(P=100):
#    t,y = make_sine(P=P, even=False)
#    a = ACF(method='scargle')
#    a.fit(t,y)
#    pbest = a.period_search()
#    assert_allclose(pbest, P, 0.01)

