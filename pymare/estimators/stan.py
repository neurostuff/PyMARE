'''Stan meta-regression estimator.'''

import numpy as np
try:
    from pystan import StanModel
except:
    StanModel = None

from .estimators import validate_input


class StanMetaRegression:
    """Bayesian meta-regression estimator using Stan.

    Args:
        sampling_kwargs: Optional keyword arguments to pass on to the MCMC
            sampler (e.g., `iter` for number of iterations).

    Notes:
        For most uses, this class should be ignored in favor of the functional
        stan() estimator. The object-oriented interface is useful primarily
        when fitting the meta-regression model repeatedly to different data;
        the separation of .compile() and .fit() steps allows one to compile
        the model only once.
    """

    def __init__(self, **sampling_kwargs):

        if StanModel is None:
            raise ImportError("Unable to import PyStan package. Is it "
                              "installed?")
        self.sampling_kwargs = sampling_kwargs
        self.model = None
        self.result_ = None

    def compile(self):
        """Compile the Stan model."""
        # Note: we deliberately use a centered parameterization for the
        # thetas at the moment. This is sub-optimal in terms of estimation,
        # but allows us to avoid having to add extra logic to detect and
        # handle intercepts in X.
        spec = f"""
        data {{
            int<lower=1> N;
            int<lower=1> K;
            vector[N] y;
            int<lower=1,upper=K> id[N];
            int<lower=1> C;
            matrix[K, C] X;
            vector[N] sigma;
        }}
        parameters {{
            vector[C] beta;
            vector[K] theta;
            real<lower=0> tau2;
        }}
        transformed parameters {{
            vector[N] mu;
            mu = theta[id] + X * beta;
        }}
        model {{
            y ~ normal(mu, sigma);
            theta ~ normal(0, tau2);
        }}
        """
        self.model = StanModel(model_code=spec)

    def fit(self, y, v, X, groups=None):
        """Run the Stan sampler and return results.

        Args:
            y (ndarray): 1d array of study-level estimates
            v (ndarray): 1d array of study-level variances
            X (ndarray): 1d or 2d array containing study-level predictors
                (including intercept); has dimensions K x P, where K is the
                number of studies and P is the number of predictor variables.
            groups ([int], optional): 1d array of integers identifying
                groups/clusters of observations in the y/v/X inputs. If
                provided, values must consist of integers in the range of 1..k
                (inclusive), where k is the number of distinct groups. When
                None (default), it is assumed that each observation in the
                inputs is a separate group.

        Returns:
            A StanFit4Model object (see PyStan documentation for details).

        Notes:
            This estimator supports (simple) hierarchical models. When multiple
            estimates are available for at least one of the studies in `y`, the
            `groups` argument can be used to specify the nesting structure
            (i.e., which rows in `y`, `v`, and `X` belong to each study).
        """
        if self.model is None:
            self.compile()

        N = y.shape[0]
        groups = groups or np.arange(1, N + 1, dtype=int)
        K = len(np.unique(groups))

        data = {
            "K": K,
            "N": N,
            'id': groups,
            'C': X.shape[1],
            'X': X,
            'y': y.ravel(),
            'sigma': v.ravel()
        }

        self.result_ = self.model.sampling(data=data, **self.sampling_kwargs)
        return self.result_


@validate_input
def stan(y, v, X, groups=None, **sampling_kwargs):
    """Fit a Bayesian meta-regression using Stan.

    Args:
        y (ndarray): 1d array of study-level estimates
        v (ndarray): 1d array of study-level variances
        X (ndarray): 1d or 2d array containing study-level predictors
            (including intercept); has dimensions K x P, where K is the number
            of studies and P is the number of predictor variables.
        groups ([int], optional): 1d array of integers identifying
            groups/clusters of observations in the y/v/X inputs. If provided,
            values must consist of integers in the range of 1..k (inclusive),
            where k is the number of distinct groups. When None (default), it
            is assumed that each observation in the inputs is a separate group.
        sampling_kwargs: Optional keyword arguments to pass on to the MCMC
            sampler (e.g., `iter` for number of iterations).

    Returns:
        A StanFit4Model object (see PyStan documentation for details).

    Notes:
        In contrast to the other meta-regression estimators, the Stan estimator
        supports estimation of (simple) hierarchical models. When multiple
        estimates are available for at least one of the studies in `y`, the
        `groups` argument can be used to specify the nesting structure (i.e.,
        which of the rows in `y`, `v`, and `X` belong to each study).
    """
    model = StanMetaRegression(**sampling_kwargs)
    return model.fit(y, v, X, groups)
