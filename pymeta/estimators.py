import numpy as np


def fixed_effects_meta_analysis(y, v):
    """ Standard inverse variance-weighted fixed-effects meta-analysis. """
    w = 1./v
    return y.dot(w) / w.sum()


def weighted_least_squares(y, v, X, tau2=0):
    """ Weighted least-squares estimation of fixed effects. """
    w = 1. / (v + tau2)
    return (np.linalg.pinv((X.T * w).dot(X)).dot(X.T) * w).dot(y)


def dersimonian_laird(y, v, X):
    """ DerSimonian-Laird random effects estimator. """
    k = len(y)
    p = X.shape[1]
    # WLS estimate of beta with tau^2 = 0
    beta_wls = weighted_least_squares(y, v, X)
    # Cochrane's Q
    w = 1. / v
    w_sum = w.sum()
    Q = (w * (y - X.dot(beta_wls))**2).sum()
    # D-L estimate of tau^2
    precision = np.linalg.pinv((X.T * w).dot(X))
    A = w_sum - np.trace((precision.dot(X.T) * w**2).dot(X))
    tau_dl = np.max([0., (Q - k + p) / A])
    # Re-estimate beta with tau^2 estimate
    beta_dl = weighted_least_squares(y, v, X, tau2=tau_dl)
    return beta_dl, tau_dl
