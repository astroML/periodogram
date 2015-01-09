"""
Autocorrelation function for models of periodic sources
"""

from __future__ import print_function,division
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import median_filter
from scipy.optimize import leastsq, curve_fit

import logging

import acor

from .findpeaks import peakdetect
from .base import PeriodicModeler



class acf(PeriodicModeler):
    """
    Autocorrelation Function for periodic modeling

    kwargs
    ------
    smooth=18

    days=True

    

    """

    def __init__(self, *args, **kwargs):

        # Set cadence (use astropy.units)
        self.cadence = 0.02043423 #d #PLACEHOLDER
        self.default_maxlag = 50//self.cadence

        #set private variables for cached acorr calculation
        self._lag = None  #always should be in cadences
        self._ac = None 

    def fit(self, t, y, dy=None, filts=None):
        """Provide data for the fit"""
        self.t = np.atleast_1d(t)
        self.y = np.atleast_1d(y)
        self.dy = dy

        mask = np.isnan(self.y)
        self.mask = mask
        self.y = self.y[~self.mask]

    def predict(self, t, filts=None, period=None):
        """Predict the best-fit model at t for the given period"""
        raise NotImplementedError()

    def score(self, period):
        """Compute the score for a period or array of periods"""
        

    def period_search(self, period=None, fit_npeaks=4,
                   smooth=18, maxlag=None, lookahead=5,
                   tol=0.2, return_peaks=False):
        """Find the best period for the model"""
        # This was acorr_period_fit

        peaks, lphs, hts = self.acorr_peaks(smooth=smooth, maxlag=maxlag,
                                            lookahead=lookahead, return_heights=True)
        logging.debug(peaks)
        logging.debug(lphs)
        logging.debug(hts)

        firstpeak = peaks[0]
### This is having trouble with the test cases (second peak is only 0.01 higher, but it's not the input
#        if lphs[0] >= lphs[1]:
#            firstpeak = peaks[0]
#        else:
#            firstpeak = peaks[1]
#            if lphs[1] < 1.2*lphs[0]:
#                logging.warning('Second peak (selected) less than 1.2x height of first peak.')

        if period is None:
            period = firstpeak

        if fit_npeaks > len(peaks):
            fit_npeaks = len(peaks)
        #peaks = peaks[:fit_npeaks]

        #identify peaks to use in fit: first 'fit_npeaks' peaks closest to integer
        # multiples of period guess

        fit_peaks = []
        fit_lphs = []
        fit_hts = []
        last = 0.
        #used = np.zeros_like(peaks).astype(bool)
        for n in np.arange(fit_npeaks)+1:
            #find highest peak within 'tol' of integer multiple (that hasn't been used)
            close = (np.absolute(peaks - n*period) < (tol*n*period)) & ((peaks-last) > 0.3*period)
            if close.sum()==0:
                fit_npeaks = n-1
                break
                #raise NoPeakError('No peak found near {}*{:.2f}={:.2f} (tol={})'.format(n,period,n*period,tol))
            ind = np.argmax(hts[close])
            last = peaks[close][ind]
            fit_peaks.append(peaks[close][ind])
            fit_lphs.append(lphs[close][ind])
            fit_hts.append(hts[close][ind])
            #used[close][ind] = True
            logging.debug('{}: {}, {}'.format(n*period,peaks[close],peaks[close][ind]))
            #logging.debug(used)

            #ind = np.argmin(np.absolute(peaks - n*period)) #closest peak
            #fit_peaks.append(peaks[ind])
            #fit_lphs.append(lphs[ind])

        logging.debug('fitting peaks: {}'.format(fit_peaks))

        if fit_npeaks < 3:
            return peaks,-1, fit_peaks, fit_lphs, fit_hts


        x = np.arange(fit_npeaks + 1)
        y = np.concatenate([np.array([0]),fit_peaks])

        #x = np.arange(fit_npeaks) + 1
        #y = fit_peaks

        def fn(x,a,b):
            return a*x + b

        fit,cov = curve_fit(fn, x, y, p0=(period,0))
        if return_peaks:
            return fit[0],cov[0][0],fit_peaks,fit_lphs,fit_hts
        else:
            return fit[0],cov[0][0]


        # TODO: implement using the score() function provided by subclasses  ############

        

    def acorr(self, maxlag=None, recalc=False, **kwargs):

        smooth = kwargs.get("smooth",18)
        days = kwargs.get("days",True)

        if maxlag is None: 
            maxlag = self.default_maxlag

        
        if self._ac is not None and not recalc:
            lag = self._lag
            ac = self._ac
        else:
            x = self.y.copy()
            x[self.mask] = 0

            #logging.debug('{} nans in x'.format((np.isnan(x)).sum()))

            ac = acor.function(x, maxlag)
            lag = np.arange(maxlag)

            #smooth AC function
            ac = gaussian_filter(ac, smooth)

            #set private variables for cached calculation
            self._ac = ac
            self._lag = lag
            self._maxlag = maxlag
            self._smooth = smooth

        if days:
            return lag*self.cadence,ac
        else:
            return lag,ac

    def acorr_peaks(self, lookahead=5, days=True, 
                    return_heights=False, **kwargs):
        lag, ac = self.acorr(**kwargs)
        return peaks_and_lphs(ac, lag, return_heights=return_heights,
                              lookahead=lookahead)
        


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

