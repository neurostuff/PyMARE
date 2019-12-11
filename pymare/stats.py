"""Miscellaneous statistical functions."""

def q_profile(y, v, X, alpha):
    """Get tau^2 CIs via the Q-Profile method (Viechtbauer, 2007)."""
    k, p = X.shape
    df = k - p
    l_crit = ss.chi2.ppf(1 - alpha / 2, df)
    u_crit = ss.chi2.ppf(alpha / 2, df)
    args = (y, X, v)
    lb = root(lambda x: (q_gen(x, *args) - l_crit)**2, 0).x[0]
    ub = root(lambda x: (q_gen(x, *args) - u_crit)**2, 100).x[0]
    return {'ci_l': lb, 'ci_u': ub}


def q_gen(tau2, y, X, v):
    from .estimators import weighted_least_squares
    beta = weighted_least_squares(y, v, X, tau2=tau2)['beta']
    w = 1. / (v + tau2)
    return (w * (y - X.dot(beta)) ** 2).sum()
