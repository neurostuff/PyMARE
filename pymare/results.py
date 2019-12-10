"""Tools for representing and manipulating meta-regression results."""

import numpy as np
import pandas as pd
from scipy.optimize import root
import scipy.stats as ss


class MetaRegressionResults:

    def __init__(self, estimator, dataset, beta, tau2, ci_method='QP',
                 alpha=0.05):
        self.estimator = estimator
        self.dataset = dataset
        self.beta = {'est': beta}
        self.tau2 = {'est': tau2}
        self.ci_method = ci_method
        self.alpha = alpha

    def summary(self):
        pass

    def plot(self):
        pass

    def to_df(self):
        fixed = self.beta.copy()
        fixed['name'] = self.dataset.X_names
        fixed = pd.DataFrame(fixed)

        tau2 = pd.DataFrame(pd.Series(self.tau2)).T
        tau2['name'] = 'tau^2'

        df = pd.concat([fixed, tau2], axis=0, sort=False)
        df = df.loc[:, ['name', 'est', 'se', 'z', 'p', 'ci_l', 'ci_u']]
        ci_l = 'ci_{:.6g}'.format(self.alpha / 2)
        ci_u = 'ci_{:.6g}'.format(1 - self.alpha / 2)
        df.columns = ['name', 'estimate', 'se', 'z-score', 'p-val', ci_l, ci_u]

        # Derived statistics

        return df

    def compute_stats(self, method=None, alpha=None):
        """Compute post-estimation stats (SE and CI) for beta and tau^2."""
        if alpha is not None:
            self.alpha = alpha
        if method is not None:
            self.ci_method = method

        self._compute_beta_stats()
        self._compute_tau2_stats()

    def _compute_beta_stats(self):
        v, X, alpha = self.dataset.v, self.dataset.X, self.alpha
        w = 1. / (v + self.tau2['est'])
        estimate = self.beta['est']
        se = np.sqrt(np.diag(np.linalg.pinv((X.T * w).dot(X))))
        z_se = ss.norm.ppf(1 - alpha / 2)
        self.beta['se'] = se
        self.beta['ci_l'] = estimate - z_se * se
        self.beta['ci_u'] = estimate + z_se * se
        self.beta['z'] = z = estimate / se
        self.beta['p'] = 1 - np.abs(0.5 - ss.norm.cdf(z)) * 2

    def _compute_tau2_stats(self):
        self._q_profile()

    def _q_profile(self):
        """Get tau^2 CIs via the Q-Profile method (Viechtbauer, 2007)."""
        dataset, alpha = self.dataset, self.alpha
        df = dataset.k - dataset.p
        l_crit = ss.chi2.ppf(1 - alpha / 2, df)
        u_crit = ss.chi2.ppf(alpha / 2, df)
        lb = root(lambda x: (q_gen(x, dataset) - l_crit)**2, 0).x[0]
        ub = root(lambda x: (q_gen(x, dataset) - u_crit)**2, 100).x[0]
        self.tau2['ci_l'] = lb
        self.tau2['ci_u'] = ub


def q_gen(tau2, dataset):
    from .estimators import WeightedLeastSquares
    beta = WeightedLeastSquares(tau2).fit(dataset).beta['est']
    v, y, X = dataset.v, dataset.y, dataset.X
    w = 1. / (v + tau2)
    return (w * (y - X.dot(beta)) ** 2).sum()
