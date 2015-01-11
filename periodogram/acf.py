"""
Autocorrelation function for models of periodic sources
"""

from __future__ import print_function,division
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline as interpolate
from scipy.ndimage.filters import gaussian_filter

import logging

import acor

from .findpeaks import peakdetect
from .base import PeriodicModeler



class ACF(PeriodicModeler):
    """
    Autocorrelation function for periodic modeling.

    This can only deal with evenly sampled data.  Use ACFUneven
    for implementation of Scargle (1989) ACF for uneven time sampling.

    """

    def __init__(self, maxlag=None, smooth=None):
        self.maxlag = maxlag
        self.smooth = smooth

        self.power_fn = None
        
    def fit(self, t, y, dy=None, filts=None):
        """Data must be evenly sampled.
        """
        
        #TODO: add real test to see if data is evenly sampled with no gaps
        # for now, just assume it is 
        
        self.t = np.atleast_1d(t)
        self.y = np.atleast_1d(y)
        self.dy = dy #ignored

        if self.maxlag is None:
            maxlag = len(self.y)
        else:
            maxlag = self.maxlag
        
        self.cadence = np.median(self.t[1:] - self.t[:-1])
        
        self.ac = acor.function(self.y, maxlag)
        if self.smooth is not None:
            ac = gaussian_filter(ac, smooth)
        
        self.lag = np.arange(maxlag) * self.cadence

        self.power_fn = interpolate(self.ac, self.lag, s=0, k=1)

    def score(self, period):
        """Compute the score for a period or array of periods"""
        if self.power_fn is None:
            raise RuntimeError('Must fit before calculating score')

        return self.power_fn(period)
                

    def period_search(self, lookahead=5):
        """Finds most promising peak; returns as best period. 
        """

        peaks, lphs, hts = peaks_and_lphs(self.ac, self.lag, return_heights=True,
                                          lookahead=lookahead)

        #Return first peak, unless second peak is more than 5% higher (arbitrary, granted)
        if lphs[1] > 1.05*lphs[0]:
            logging.warning('ACF period_search: second peak ({0[1]}) is more than 5% higher than first ({0[0]}); using second'.format(lphs))
            return peaks[1]

        else:
            return peaks[0]
        
        


def peaks_and_lphs(y, x=None, lookahead=5, return_heights=False):
    """Returns locations of peaks and corresponding "local peak heights"
    """
    if x is None:
        x = np.arange(len(y))

    maxes, mins = peakdetect(y, x, lookahead=lookahead)
    maxes = np.array(maxes)
    mins = np.array(mins)
    
    #logging.debug('maxes: {}'.format(maxes))

    #calculate "local heights".  First will always be a minimum.
    try: #this if maxes and mins are same length 
        lphs = np.concatenate([((maxes[:-1,1] - mins[:-1,1]) + (maxes[:-1,1] - mins[1:,1]))/2.,
                               np.array([maxes[-1,1]-mins[-1,1]])])
    except ValueError: #this if mins have one more
        lphs = ((maxes[:,1] - mins[:-1,1]) + (maxes[:,1] - mins[1:,1]))/2.

    if return_heights:
        return maxes[:,0], lphs, maxes[:,1]
    else:
        return maxes[:,0], lphs 

