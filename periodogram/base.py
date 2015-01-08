"""
Base class for models of periodic sources
"""


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

    def period_search(self):
        """Find the best period for the model"""
        # TODO: implement using the score() function provided by subclasses

class AperiodicModeler(object):
    """Base class for periodic modeling"""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("AperiodicModeler")

    def fit(self, t, y, dy=None, filts=None):
        """Provide data for the fit"""
        raise NotImplementedError()