from __future__ import division,print_function
"""
Base class for time series
"""

import numpy as np

class TimeSeries(self, t, f, df=None, mask=None):
    """
    Base clase for time series data

    Parameters
    ----------
    t : array_like
        times

    f : array_like
        fluxes, must be same length as times

    df : float array_like, optional
        uncertainties; if float, then all assumed to be the same;
        

    mask : array_like or 
    """

    assert(t.shape == f.shape)
    
    self.t = t
    self.f = f
    if df is not None:
        if np.size(df)==1:
            df = np.ones_like(f) * df
        else:
            assert(df.shape == f.shape)
    self.df = df

    if mask is None:
        mask = np.isnan(f)
    self.mask = mask
