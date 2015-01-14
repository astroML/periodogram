"""
Autocorrelation function for models of periodic sources
"""

from __future__ import print_function,division
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline as interpolate
from scipy.ndimage.filters import gaussian_filter, median_filter

import logging

import acor
from astroML.time_series import ACF_scargle, ACF_EK

from .findpeaks import peakdetect
from .base import PeriodicModeler

class ACF(PeriodicModeler):
    """
    Autocorrelation function for periodic modeling.

    Requires evenly sampled data.  For non evenly sampled data,
    use ACF_scargle or ACF_EK.

    Parameters
    ----------

    maxlag : float (optional)
        Maximum period to search for.  If not provided, then will
        default to 1/2 data span.

    method : 'standard', 'scargle' or 'EK' (optional)
        Method to use to calculate ACF.  If standard is chosen
        then data must be evenly sampled.

    smooth : float (optional)
        Timescale over which to smooth ACF. 
        For 'standard' or 'EK' method, this will run a gaussian filter with
        given width over the acf; for 'scargle' it will first median-smooth,
        then gaussian-smooth.

    n_omega, omega_max : int, float (optional)
        Passed to ``astroML.time_series.ACF_scargle``

    bins : int or array_like (optional)
        Passed to ``astroML.time_series.ACF_EK``; bins in which to calculate
        ACF
    """

    def __init__(self, maxlag=None, method='standard',
                 smooth=None,
                 n_omega=2**12, omega_max=100, bins=None):
        
        self.maxlag = maxlag
        self.smooth = smooth

        if method not in ['standard','scargle','EK']:
            raise ValueError('Unrecognized method {}'.format(method))
        self.method = method
        
        if self.method=='scargle':
            self.n_omega = n_omega
            self.omega_max = omega_max
        elif self.method=='EK':
            self.bins = bins
                        
        self.power_fn = None
        
    def fit(self, t, y, dy=None):
        """Data must be evenly sampled.
        """

        self.t = np.atleast_1d(t)
        self.y = np.atleast_1d(y)
        self.dy = dy #ignored

        #set default max lag to be 1/2 total span of observation, if not otherwise set
        if self.maxlag is None:
            maxlag = (self.t[-1] - self.t[0])/2.
        else:
            maxlag = self.maxlag
        
        if self.method=='standard':
            #TODO: add test to make sure data is evenly sampled
            
            self.cadence = np.median(self.t[1:] - self.t[:-1])

            n_maxlag = int(maxlag/self.cadence)
            self.ac = acor.function(self.y, n_maxlag)
            self.lag = np.arange(n_maxlag) * self.cadence
            
        elif self.method=='scargle':
            if dy is None:
                dy = 1

            ac, lag = ACF_scargle(t, y, dy,
                                  n_omega=self.n_omega,
                                  omega_max=self.omega_max)
            #print(ac,lag)
            ind = (lag >= 0) & (lag <= maxlag)
            self.ac = ac[ind]
            self.lag = lag[ind]
            
        elif self.method=='EK':
            if dy is None:
                dy = 1

            if self.bins is None:
                bins = np.linspace(0, maxlag, 500)
            else:
                bins = self.bins
            ac, ac_err, bins = ACF_EK(t, y, dy, bins=bins)           
            
            lag = 0.5 * (bins[1:] + bins[:-1])
            ind = (lag >=0) & (lag <= maxlag)
            self.ac = ac[ind]
            self.lag = lag[ind]
            self.ac_err = ac_err[ind]
                        
        if self.smooth is not None:
            if self.method == 'EK':
                dlag = self.lag[1]-self.lag[0] #assuming evenly spaced bins
                kern_width = self.smooth//dlag
                self.ac = gaussian_filter(self.ac, kern_width)
            if self.method=='standard':
                kern_width = self.smooth / self.cadence
                self.ac = gaussian_filter(self.ac, kern_width)
            if self.method=='scargle':
                #median smooth first, because you can get crazy spikes,
                # then gaussian-smooth
                dlag = self.lag[1]-self.lag[0] #assuming evenly spaced bins
                kern_width = self.smooth//dlag 
                self.ac = median_filter(self.ac, kern_width)
                self.ac = gaussian_filter(self.ac, kern_width)
                
            

        try:
            self.power_fn = interpolate(self.lag, self.ac, s=0, k=1)
        except:
            print('Error generating interpolation function: {}, {}'.format(self.lag, self.ac))
        
        return self
        
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

