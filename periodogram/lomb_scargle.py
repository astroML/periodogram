from astroML.time_series import multiterm_periodogram
from scipy.interpolate import UnivariateSpline as interpolate
import numpy as np

from .base import PeriodicModeler

class LombScargle(PeriodicModeler):
    """Class implementing Lomb-Scargle fitting"""
    def __init__(self, frequencies=None, pmin=None, pmax=None, 
            resolution=1000, linspace=False):
        
        if frequencies is not None:
            self.frequencies = frequencies
            self.periods = 2*np.pi/self.frequencies
        else:
            if pmin is None or pmax is None or resolution is None:
                raise ValueError('Must provide either frequencies or min, max, resolution')

            if linspace:
                self.periods = np.linspace(pmin, pmax, resolution)
            else:
                self.periods = np.logspace(np.log10(pmin),
                                        np.log10(pmax),
                                        resolution)
            self.frequencies = 2*np.pi/self.periods
        self.power = None
        self.power_fn = None
    
    def fit(self, t, y, dy=None, filts=None):
        """Provide data for the fit"""
        if dy is None:
            dy = np.ones_like(y)
        if np.size(dy)==1:
            dy = np.ones_like(y)*dy
        
        self.powers = multiterm_periodogram(t, y, dy, self.frequencies,
                                            n_terms=1)
        self.power_fn = interpolate(self.periods, self.powers, s=0, k=1)
        return self

    def score(self, period):
        """Compute the score for a period or array of periods"""
        if self.power_fn is None:
            raise RuntimeError("Need to fit data first.")
        else:
            return self.power_fn(period)
    
    def period_search(self, pmin=None, pmax=None, resolution=None, **kwargs):
        """Find the best period for the model"""
        if pmin is None: 
            pmin = self.periods.min()
        if pmax is None:
            pmax = self.periods.max()
        if resolution is None:
            resolution = len(self.periods)
        return super(LombScargle,self).period_search(pmin, pmax, 
                    resolution=resolution, **kwargs)
