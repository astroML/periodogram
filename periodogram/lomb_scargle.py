from astroML.time_series import multiterm_periodogram
from scipy.interpolate import UnivariateSpline as interpolate
import numpy as np        

from .base import PeriodicModeler

class LombScargle(PeriodicModeler):
    """Class implementing Lomb-Scargle fitting"""
    def __init__(self, frequencies=None, min_period=None, max_period=None, 
            num_periods=None):
        if frequencies is not None:
            self.frequencies = frequencies
            self.periods = 2*np.pi/self.frequencies
        else:
            self.periods = np.logspace(np.log10(min_period),
                    np.log_10(max_period), num_periods)
            self.frequencies = 2*np.pi/self.periods
        self.power = None
        self.power_fn = None
    
    def fit(self, t, y, dy=None, filts=None):
        """Provide data for the fit"""
        self.powers = multiterm_periodogram(t, y, dy, self.frequencies,
                                            nterms=1)
        self.power_fn = interpolate(self.periods, self.powers, s=0, k=0)

    def score(self, period):
        """Compute the score for a period or array of periods"""
        if self.power_fn is None:
            raise RuntimeError("Need to fit data first.")
        else:
            return self.power_fn(period)

