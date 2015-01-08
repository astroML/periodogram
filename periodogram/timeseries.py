from __future__ import division,print_function
"""
Base class for time series
"""

import numpy as np
import matplotlib.pyplot as plt

class TimeSeries(object):
    def init(self, t, f, df=None, mask=None,
             band=None):
        """
        Base class for time series data

        Parameters
        ----------
        t : array_like
            times

        f : array_like
            fluxes, must be same length as times

        df : float array_like, optional
            uncertainties; if float, then all assumed to be the same;
            if array, then must be same length as times and fluxes

        mask : array_like or None

        band : string or None
            Passband that data was taken in.
        """

        assert(t.shape == f.shape)

        self._t = t
        self._f = f
        if df is not None:
            if np.size(df)==1:
                df = np.ones_like(f) * df
            else:
                assert(df.shape == f.shape)
        self._df = df

        if mask is None:
            mask = np.isnan(f)
        self._mask = mask

        self.band = band

        self.models = []

    @property
    def t(self):
        return self._t[~self._mask]

    @property
    def f(self):
        return self._f[~self._mask]

    @property
    def df(self):
        return self._df[~self._mask]
        
    def add_perodic_model(self, model, *args, **kwargs):
        """Connects and fits PeriodicModeler object

        Parameters
        ----------
        model: PeriodicModeler, or string
            PeriodicModeler object or string indicating known PeriodicModel

        args, kwargs passed on to PeriodicModeler
        """
        m = model(*args,**kwargs)
        m.fit(self.t, self.f, self.df)

        self.models.append(m)

    def plot(self, **kwargs):
        plt.plot(self.t, self.f, **kwargs)

    
