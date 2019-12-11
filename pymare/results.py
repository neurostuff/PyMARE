"""Tools for representing and manipulating meta-regression results."""

import numpy as np
import pandas as pd
from scipy.optimize import root
import scipy.stats as ss

from .stats import q_profile, q_gen


class MetaRegressionResults:

    def __init__(self, params, dataset, ci_method='QP', alpha=0.05):
        self.params = {name:{'est': val} for name, val in params.items()}
        self.dataset = dataset
        self.ci_method = ci_method
        self.alpha = alpha

    def __getitem__(self, key):
        return self.params[key]

    def summary(self):
        pass

    def plot(self):
        pass

    def to_df(self):
        fixed = self.params['beta'].copy()
        fixed['name'] = self.dataset.names
        fixed = pd.DataFrame(fixed)

        tau2 = pd.DataFrame(pd.Series(self.params['tau2'])).T
        tau2['name'] = 'tau^2'

        df = pd.concat([fixed, tau2], axis=0, sort=False)
        df = df.loc[:, ['name', 'est', 'se', 'z', 'p', 'ci_l', 'ci_u']]
        ci_l = 'ci_{:.6g}'.format(self.alpha / 2)
        ci_u = 'ci_{:.6g}'.format(1 - self.alpha / 2)
        df.columns = ['name', 'estimate', 'se', 'z-score', 'p-val', ci_l, ci_u]

        return df

    def compute_stats(self, method=None, alpha=None):
        """Compute post-estimation stats (SE and CI) for beta and tau^2."""
        if alpha is not None:
            self.alpha = alpha
        if method is not None:
            self.ci_method = method

        def _compute_beta_stats(self):
            v, X, alpha = self.dataset.variances, self.dataset.predictors, self.alpha
            w = 1. / (v + self['tau2']['est'])
            estimate = self['beta']['est']
            se = np.sqrt(np.diag(np.linalg.pinv((X.T * w).dot(X))))
            z_se = ss.norm.ppf(1 - alpha / 2)
            z = estimate / se

            self['beta'].update({
                'se': se,
                'ci_l': estimate - z_se * se,
                'ci_u': estimate + z_se * se,
                'z': z,
                'p': 1 - np.abs(0.5 - ss.norm.cdf(z)) * 2
            })

        def _compute_tau2_stats(self):
            y, v, X = self.dataset.y, self.dataset.v, self.dataset.X
            alpha = self.alpha
            ci = q_profile(y, v, X, alpha)
            self['tau2'].update(ci)

        self._compute_beta_stats()
        self._compute_tau2_stats()
