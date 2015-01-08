import numpy as np
import scipy.optimize

class Variogram(object):
    """ Base Class for Variogram (aka Structure Function)"""
    def __init__(self):
        pass
        #raise NotImplementedError()
    def calc_variogram(self, t, y, dy=0.0):
    	"""
        Parameters
        ----------
        t : array_like
            time locations of the points, in days
        y : array_like
            y locations of the points
        dy : array_like or float (default = 1)
            Errors in the y values
        
        Returns
        -------
        delta_t : array_like
             time steps in years
        delta_y : array_like
             y location changes
        delta_t : array_like
             errors in the y location changes
        """
        delta_t = []
        delta_y = []
        delta_dy = []
        n = len(t)
        for j in range(n):
            for k in range(n):
                if j > k:
                    delta_t.append( abs( t[j] - t[k])/365.242199 )
                    delta_y.append( abs( y[j] - y[k]) )
                    delta_dy.append( scipy.sqrt( dy[j]**2. + dy[k]**2.) )
        return delta_t, delta_y, delta_dy

class PowerLaw(Variogram):
    """
    Power Law fit to the structure function
    
    V = A*(delta_t)**(gamma)"""
    def __init__(self):
        pass
        #raise NotImplementedError()
    def likelihoodfunc(self, p):
        summation = np.array([])
        for i in range(len(self.delta_t)):
            summation = np.append(summation, np.log(((p[0]*self.delta_t[i]**p[1])**2. + self.delta_dy[i]**2.)) + ((self.delta_y[i]**2.)/((p[0]*self.delta_t[i]**p[1])**2. + self.delta_dy[i]**2.))) #power-law fit
        if (p[0] < 0.0) or (p[1] < 0.0) or (len(summation[np.isfinite(summation)]) != len(summation)):
            return 10.**32.
        else:
            return (np.sum(summation)/len(summation))
    def fit(self, t, y, dy):
        self.delta_t, self.delta_y, self.delta_dy = self.calc_variogram(t, y, dy)
        pinit_likelihood = [0.1, 0.1]
        xopt, fopt, direc, iter, funcalls, warnflag, allvecs = scipy.optimize.fmin_powell(self.likelihoodfunc, pinit_likelihood, full_output=1, disp=0, retall=1)
        if fopt == 10.**32.:
            A_likelihood = np.nan
            gamma_likelihood = np.nan
        else:
            A_likelihood = xopt[0]
            gamma_likelihood = xopt[1]
        return A_likelihood, gamma_likelihood

class DRW(Variogram):
    """
    Damped Random Walk fit to the structure function
    
    V = SFinf*(1 - exp(-delta_t/tau))**(1/2)"""
    def __init__(self):
        pass
        #raise NotImplementedError()
    def likelihoodfunc(self, p):
        summation = np.array([])
        for i in range(len(self.delta_t)):
            summation = np.append(summation, np.log((p[0]**2.)*(1.-np.exp(-self.delta_t[i]/p[1])) + self.delta_dy[i]**2.) + ((self.delta_y[i]**2.)/((p[0]**2.)*(1.-np.exp(-self.delta_t[i]/p[1])) + self.delta_dy[i]**2.))) #DRW fit
        if (p[0] < 0.0) or (p[1] < 0.0) or (len(summation[np.isfinite(summation)]) != len(summation)):
            return 10.**32.
        else:
            return (np.sum(summation)/len(summation))
    def fit(self, t, y, dy):
    	self.delta_t, self.delta_y, self.delta_dy = self.calc_variogram(t, y, dy)
        pinit_likelihood = [0.1, 0.1]
        xopt, fopt, direc, iter, funcalls, warnflag, allvecs = scipy.optimize.fmin_powell(self.likelihoodfunc, pinit_likelihood, full_output=1, disp=0, retall=1)
        if fopt == 10.**32.:
            tau_likelihood = np.nan
            SFinf_likelihood = np.nan
        else:
            tau_likelihood = xopt[0]
            SFinf_likelihood = xopt[1]
        return tau_likelihood, SFinf_likelihood