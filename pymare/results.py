"""Tools for representing and manipulating meta-regression results."""
from functools import lru_cache
from warnings import warn
import itertools
from inspect import getfullargspec

import numpy as np
import pandas as pd
import scipy.stats as ss

try:
    import arviz as az
except:
    az = None

from .stats import q_profile, q_gen


class MetaRegressionResults:
    """Container for results generated by PyMARE meta-regression estimators.

    Args:
        estimator (`pymare.estimators.BaseEstimator`): The estimator used to
            produce the results.
        dataset (`pymare.Dataset`): A Dataset instance containing the inputs
            to the estimator.
        fe_params (NDarray): Fixed-effect coefficients. Must be a 2-d numpy
            array with shape p x d, where p is the number of predictors, and
            d is the number of parallel datasets (typically 1).
        fe_cov (NDArray): The p x p inverse covariance (or precision) matrix
            for the fixed effects.
        tau2 (NDArray, float, optional): A 1-d array containing the estimated
            tau^2 value for each parallel dataset (or a float, for a single
            dataset). May be omitted by fixed-effects estimators.
    """
    def __init__(self, estimator, dataset, fe_params, fe_cov, tau2=None):
        self.estimator = estimator
        self.dataset = dataset
        self.fe_params = fe_params
        self.fe_cov = fe_cov
        self.tau2 = tau2

    @property
    @lru_cache(maxsize=1)
    def fe_se(self):
        cov = np.atleast_3d(self.fe_cov) # 3rd dim is for parallel datasets
        return np.sqrt(np.diagonal(cov)).T

    @lru_cache(maxsize=16)
    def get_fe_stats(self, alpha=0.05):

        beta, se = self.fe_params, self.fe_se
        z_se = ss.norm.ppf(1 - alpha / 2)
        z = beta / se

        stats = {
            'est': beta,
            'se': se,
            'ci_l': beta - z_se * se,
            'ci_u': beta + z_se * se,
            'z': z,
            'p': 1 - np.abs(0.5 - ss.norm.cdf(z)) * 2
        }

        return stats

    @lru_cache(maxsize=16)
    def get_re_stats(self, method='QP', alpha=0.05):
        if method == 'QP':
            n_iters = np.atleast_2d(self.tau2).shape[1]
            if n_iters > 10:
                warn("Method 'QP' is not parallelized; it may take a while to "
                     "compute CIs for {} parallel tau^2 values.".format(n_iters))

            # Make sure we have an estimate of v if it wasn't observed
            v = self.estimator.get_v(self.dataset)

            cis = []
            for i in range(n_iters):
                args = {
                    'y': self.dataset.y[:, i],
                    'v': v[:, i],
                    'X': self.dataset.X,
                    'alpha': alpha,
                }

                try:
                    q_cis = q_profile(**args)
                except Exception as exc:
                    q_cis = {'ci_l': np.nan, 'ci_u': np.nan}

                cis.append(q_cis)

        else:
            raise ValueError("Invalid CI method '{}'; currently only 'QP' is "
                             "available.".format(method))

        return {
            'tau^2': self.tau2,
            'ci_l': np.array([ci['ci_l'] for ci in cis]),
            'ci_u': np.array([ci['ci_u'] for ci in cis])
        }

    @lru_cache(maxsize=16)
    def get_heterogeneity_stats(self):
        v = self.estimator.get_v(self.dataset)
        q_fe = q_gen(self.dataset.y, v, self.dataset.X, 0)
        df = self.dataset.y.shape[0] - self.dataset.X.shape[1]
        i2 =np.maximum(100. * (q_fe - df) / q_fe, 0.)
        h = np.maximum(np.sqrt(q_fe / df), 1.)
        p = ss.chi2.sf(q_fe, df)
        return {
            'Q': q_fe,
            'p(Q)': p,
            'I^2': i2,
            'H': h
        }

    def to_df(self, alpha=0.05):
        """Return a pandas DataFrame summarizing fixed effect results."""
        b_shape = self.fe_params.shape
        if len(b_shape) > 1 and b_shape[1] > 1:
            raise ValueError("More than one set of results found! A summary "
                             "table cannot be displayed for multidimensional "
                             "results at the moment.")
        fe_stats = self.get_fe_stats(alpha).items()
        df = pd.DataFrame({k: v.ravel() for k, v in fe_stats})
        df['name'] = self.dataset.X_names
        df = df.loc[:, ['name', 'est', 'se', 'z', 'p', 'ci_l', 'ci_u']]
        ci_l = 'ci_{:.6g}'.format(alpha / 2)
        ci_u = 'ci_{:.6g}'.format(1 - alpha / 2)
        df.columns = ['name', 'estimate', 'se', 'z-score', 'p-value', ci_l, ci_u]
        return df

    def permutation_test(self, n_perm=1000):
        """Run permutation test.

        Args:
            n_perm (int):Number of permutations to generate. The actual number
                used may be smaller in the event of an exact test (see below),
                but will never be larger.

        Returns:
            An instance of class PermutationTestResults.

        Notes:
            If the number of possible permutations is smaller than n_perm, an
            exact test will be conducted. Otherwise an approximate test will be
            conducted by randomly shuffling the outcomes n_perm times (or, for
            intercept-only models, by randomly flipping their signs). Note that
            for closed-form estimators (e.g., 'DL' and 'HE'), permuted datasets
            are estimated in parallel. This means that one can often set very
            high n_perm values (e.g., 100k) with little performance degradation.
        """
        n_obs, n_datasets = self.dataset.y.shape
        has_mods = self.dataset.X.shape[1] > 1

        fe_stats = self.get_fe_stats()
        re_stats = self.get_re_stats()

        # create results arrays
        fe_p = np.zeros_like(self.fe_params)
        rfx = (self.tau2 is not None)
        tau_p = np.zeros((n_datasets,)) if rfx else None

        # Calculate # of permutations and determine whether to use exact test
        if has_mods:
            n_exact = np.math.factorial(n_obs)
        else:
            n_exact = 2**n_obs
            if n_exact < n_perm:
                perms = np.array(list(itertools.product([-1, 1], repeat=n_obs))).T

        exact = n_exact < n_perm
        if exact:
            n_perm = n_exact

        # Loop over parallel datasets
        for i in range(n_datasets):

            y = self.dataset.y[:, i]
            y_perm = np.repeat(y[:, None], n_perm, axis=1)

            # for v, we might actually be working with n, depending on estimator
            has_v = 'v' in getfullargspec(self.estimator.fit).args[1:]
            v = self.dataset.v[:, i] if has_v else self.dataset.n[:, i]

            v_perm = np.repeat(v[:, None], n_perm, axis=1)

            if has_mods:
                if exact:
                    perms = itertools.permutations(range(n_obs))
                    for j, inds in enumerate(perms):
                        inds = np.array(inds)
                        y_perm[:, j] = y[inds]
                        v_perm[:, j] = v[inds]
                else:
                    for j in range(n_perm):
                        np.random.shuffle(y_perm[:, j])
                        np.random.shuffle(v_perm[:, j])
            else:
                if exact:
                    y_perm *= perms
                else:
                    signs = np.random.choice(np.array([-1, 1]), (n_obs, n_perm))
                    y_perm *= signs

            # Pass parameters, remembering that v may actually be n
            kwargs = {'y': y_perm, 'X': self.dataset.X}
            kwargs['v' if has_v else 'n'] = v_perm
            params = self.estimator.fit(**kwargs).params_

            fe_obs = fe_stats['est'][:, i]
            if fe_obs.ndim == 1:
                fe_obs = fe_obs[:, None]
            fe_p[:, i] = (np.abs(fe_obs) < np.abs(params['fe_params'])).mean(1)
            if rfx:
                abs_obs = np.abs(re_stats['tau^2'][i])
                tau_p[i] = (abs_obs < np.abs(params['tau2'])).mean()

        # p-values can't be smaller than 1/n_perm
        params = {'fe_p': np.maximum(1 / n_perm, fe_p)}
        if rfx:
            params['tau2_p'] = np.maximum(1 / n_perm, tau_p)

        return PermutationTestResults(self, params, n_perm, exact)


class CombinationTestResults:
    """Container for results generated by p-value combination methods.

    Args:
        estimator (`pymare.estimators.BaseEstimator`): The estimator used to
            produce the results.
        dataset (`pymare.Dataset`): A Dataset instance containing the inputs
            to the estimator.
        z (NDArray, optional): Array of z-scores.
        p (NDArray, optional): Array of right-tailed p-values.
    """
    def __init__(self, estimator, dataset, z=None, p=None):
        self.estimator = estimator
        self.dataset = dataset
        if p is None and z is None:
            raise ValueError("One of 'z' or 'p' must be provided.")
        self._z = z
        self._p = p

    @property
    @lru_cache(maxsize=1)
    def z(self):
        if self._z is None:
            self._z = ss.norm.isf(self.p)
        return self._z

    @property
    @lru_cache(maxsize=1)
    def p(self):
        if self._p is None:
            self._p = ss.norm.sf(self.z)
        return self._p

    def permutation_test(self, n_perm=1000):
        """Run permutation test.

        Args:
            n_perm (int): Number of permutations to generate. The actual number
                used may be smaller in the event of an exact test (see below),
                but will never be larger.

        Returns:
            An instance of class PermutationTestResults.

        Notes:
            If the number of possible permutations is smaller than n_perm, an
            exact test will be conducted. Otherwise an approximate test will be
            conducted by randomly shuffling the outcomes n_perm times (or, for
            intercept-only models, by randomly flipping their signs). Permuted
            datasets are processed in parallel. This means that one can often
            set very high n_perm values (e.g., 100k) with little performance
            degradation.
        """
        n_obs, n_datasets = self.dataset.y.shape

        # create results arrays
        p_p = np.zeros_like(self.z)

        # Calculate # of permutations and determine whether to use exact test
        n_exact = 2**n_obs
        if n_exact < n_perm:
            perms = np.array(list(itertools.product([-1, 1], repeat=n_obs))).T
            exact = True
            n_perm = n_exact
        else:
            exact = False

        # Initialize a copy of the estimator to prevent overwriting results
        est = self.estimator.__class__(mode=self.estimator.mode)

        # Loop over parallel datasets
        for i in range(n_datasets):

            y = self.dataset.y[:, i]
            y_perm = np.repeat(y[:, None], n_perm, axis=1)

            if exact:
                y_perm *= perms
            else:
                signs = np.random.choice(np.array([-1, 1]), (n_obs, n_perm))
                y_perm *= signs

            # Some combination tests can handle weights (passed as v)
            kwargs = {'z': y_perm}
            if 'w' in getfullargspec(est.fit).args:
                kwargs['w'] = self.dataset.v
            params = est.fit(**kwargs).params_

            p_obs = self.z[i]
            if p_obs.ndim == 1:
                p_obs = p_obs[:, None]
            p_p[i] = (p_obs > params['p']).mean()

        # p-values can't be smaller than 1/n_perm
        p_p = np.maximum(1 / n_perm, p_p)

        return PermutationTestResults(self, {'fe_p': p_p}, n_perm, exact)


class PermutationTestResults:
    """Lightweight container to hold and display permutation test results."""
    def __init__(self, results, perm_p, n_perm, exact=False):
        self.results = results
        self.perm_p = perm_p
        self.n_perm = n_perm
        self.exact = exact

    def to_df(self, **kwargs):
        """Export permutation test results as a pandas DF.

        Args:
            kwargs: Keyword arguments to pass onto to_df() calls of parent
                results class (e.g., in case of MetaRegressionResults class,
                `alpha` is available).

        Returns:
            A pandas DataFrame that adds columns to the standard fixed effect
            result table based on permutation test results. A column is added
            for every name found in both the parent DF and the params
            dictionary passed at initialization.
        """
        df = self.results.to_df(**kwargs)
        c_ind = list(df.columns).index('p-value')
        df.insert(c_ind + 1, 'p-value (perm.)', self.perm_p['fe_p'])
        return df


class BayesianMetaRegressionResults:
    """Container for MCMC sampling-based PyMARE meta-regression estimators.

    Args:
        data (`StanFit4Model` or `InferenceData`): Either a StanFit4Model
            instanced returned from PyStan or an ArviZ InferenceData instance.
        dataset (`pymare.Dataset`): A Dataset instance containing the inputs
            to the estimator.
        ci (float, optional): Desired width of highest posterior density (HPD)
            interval. Defaults to 95%.
    """
    def __init__(self, data, dataset, ci=95.):
        if az is None:
            raise ValueError("ArviZ package must be installed in order to work"
                             " with the BayesianMetaRegressionResults class.")
        if data.__class__.__name__ == 'StanFit4Model':
            data = az.from_pystan(data)
        self.data = data
        self.dataset = dataset
        self.ci = ci

    def summary(self, include_theta=False, **kwargs):
        """Summarize the posterior estimates via ArviZ.

        Args:
            include_theta (bool, optional): Whether or not to include the
                estimated group-level means in the summary. Defaults to False.
            kwargs: Optional keyword arguments to pass onto ArviZ's summary().

        Returns:
            A pandas DataFrame, unless the `fmt="xarray"` argument is passed in
            kwargs, in which case an xarray Dataset is returned.
        """
        var_names = ['beta', 'tau2']
        if include_theta:
            var_names.append('theta')
        var_names = kwargs.pop('var_names', var_names)
        return az.summary(self.data, var_names, **kwargs)

    def plot(self, kind='trace', **kwargs):
        """Generate various plots of the posterior estimates via ArviZ.

        Args:
            kind (str, optional): The type of ArviZ plot to generate. Can be
                any named function of the form "plot_{}" in the ArviZ
                namespace (e.g., 'trace', 'forest', 'posterior', etc.).
                Defaults to 'trace'.
            kwargs: Optional keyword arguments passed onto the corresponding
                ArviZ plotting function (see ArviZ docs for details).

        Returns:
            A matplotlib or bokeh object, depending on plot kind and kwargs.
        """
        name = 'plot_{}'.format(kind)
        plotter = getattr(az, name)
        if plotter is None:
            raise ValueError("ArviZ has no plotting function '{}'.".format(name))
        plotter(self.data, **kwargs)
