"""
Base class for models of periodic sources
"""

from .findpeaks import peakdetect

class PeriodicModeler(object):
    """Base class for periodic modeling"""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("PeriodicModeler")

    def fit(self, t, y, dy=None, filts=None):
        """Provide data for the fit"""
        raise NotImplementedError()

    def predict(self, t, filts=None, period=None):
        """Predict the best-fit model at t for the given period"""
        raise NotImplementedError()

    def score(self, period):
        """Compute the score for a period or array of periods"""
        raise NotImplementedError()

    def period_search(self, pmin, pmax, resolution=1e4, nperiods=1,
                      return_scores=False, linspace=False):
        """Find the best period for the model"""
        # TODO: implement using the score() function provided by subclasses
        if linspace:
            periods = np.linspace(pmin, pmax, resolution)
        else:
            periods = np.logspace(np.log10(pmin),
                                np.log10(pmax),
                                resolution)
        
        scores = self.score(periods)
        maxes,mins = peakdetect(scores, periods)
        maxes = np.array(maxes)
        inds = np.argsort(maxes[:,1])
        
        pks,hts = maxes[inds,0][-nperiods:][::-1],maxes[inds,1][-nperiods:][::-1]

        if len(pks)==1:
            pks = pks[0]
            hts = hts[0]
        
        if return_scores:
            return pks, hts
        else:
            return pks

class AperiodicModeler(object):
    """Base class for periodic modeling"""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("AperiodicModeler")

    def fit(self, t, y, dy=None, filts=None):
        """Provide data for the fit"""
        raise NotImplementedError()