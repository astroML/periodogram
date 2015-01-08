from __future__ import print_function, division

"""
Supersmoother code for periodic modeling
"""
import numpy as np


try:
    import supersmoother as ssm
except ImportError:
    raise ImportError("Package supersmoother is required. "
                      "Use ``pip install supersmoother`` to install")


from .base import PeriodicModeler


class SuperSmoother(PeriodicModeler):
    def __init__(self):
        pass

    def fit(self, t, y, dy=1, filts=None):
        """Fit the supersmoother model to the data"""
        if filts is not None:
            raise NotImplementedError("``filts`` keyword is not supported")
        t, y, dy = np.broadcast_arrays(t, y, dy)
        self.t, self.y, self.dy = t, y, dy

        # TODO: this should actually be a weighted median, probably...
        mu = np.sum(y / dy ** 2) / np.sum(1 / dy ** 2)
        self.baseline_err = np.mean(abs((y - mu) / dy))
        return self

    def predict(self, t, filts=None, period=None):
        """Predict the supersmoother model for the data"""
        if period is None:
            raise ValueError("Must provide a period for the prediction")
        model = ssm.SuperSmoother().fit(self.t % period, self.y, self.dy)
        return model.predict(t % period)

    def score(self, period):
        """Find the score for a given period."""
        period = np.asarray(period)

        # double-up the data to allow periodicity on the fits
        N = len(self.t)
        N4 = N // 4
        t = np.concatenate([self.t, self.t])
        y = np.concatenate([self.y, self.y])
        dy = np.concatenate([self.dy, self.dy])

        results = []
        for p in period.ravel():
            # compute doubled phase and sort
            phase = t % p
            phase[N:] += p
            isort = np.argsort(phase)[N4: N + 3 * N4]
            phase = phase[isort]
            yp = y[isort]
            dyp = dy[isort]

            # compute model
            model = ssm.SuperSmoother().fit(phase, yp, dyp, presorted=True)

            # take middle residuals
            resids = model.cv_residuals()[N4: N4 + N]
            results.append(1 - np.mean(np.abs(resids)) / self.baseline_err)

        return np.asarray(results).reshape(period.shape)
        
