import numpy as np


def fixed_effects(y, v):
    """ Standard inverse variance-weighted fixed-effects meta-analysis. """
    w = 1./v
    return y.dot(w) / w.sum()  


def dersimonian_laird(y, v):
    """ DerSimonian-Laird random effects estimator. """
    k = len(y)
    # fixed-effects estimate of mu
    w = 1. / v
    w_sum = w.sum()
    mu_fe = y.dot(w) / w_sum
    # Cochrane's Q
    Q = (w * (y - mu_fe)**2).sum()
    tau_dl = np.max([0., (Q - k + 1) / (w_sum - ((w**2).sum() / w_sum))])
    return mu_fe, tau_dl
