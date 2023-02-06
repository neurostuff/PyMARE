"""Core classes and functions."""

from functools import partial

import numpy as np
import pandas as pd

from pymare.utils import _check_inputs_shape, _listify

from .estimators import (
    DerSimonianLaird,
    Hedges,
    SampleSizeBasedLikelihoodEstimator,
    StanMetaRegression,
    VarianceBasedLikelihoodEstimator,
    WeightedLeastSquares,
)
from .stats import ensure_2d


class Dataset:
    """Container for input data and arguments to estimators.

    Parameters
    ----------
    y : None or :obj:`numpy.ndarray` of shape (K,) or :obj:`str`, optional
        1d array of study-level estimates with length K, or the name of the column in data
        containing the y values.
        Default = None.
    v : None or :obj:`numpy.ndarray` of shape (K,) or :obj:`str`, optional
        1d array of study-level variances with length K, or the name of the column in data
        containing v values.
        Default = None.
    X : None or :obj:`numpy.ndarray` of shape (K,[P]) or :obj:`list` of :obj:`str`, optional
        1d or 2d array containing study-level predictors (dimensions K x P),
        or a list of strings giving the names of the columns in data containing the X values.
        Default = None.
    n : None or :obj:`numpy.ndarray` of shape (K,) or :obj:`str`, optional
        1d array of study-level sample sizes (length K), or the name of the corresponding column
        in ``data``.
        Default = None.
    data : None or :obj:`pandas.DataFrame`, optional
        A pandas DataFrame containing y, v, X, and/or n values.
        By default, columns are expected to have the same names as arguments
        (e.g., the y values will be expected in the 'y' column).
        This can be modified by passing strings giving column names to any of the ``y``, ``v``,
        ``X``, or ``n`` arguments.
        Default = None.
    X_names : None or :obj:`list` of :obj:`str`, optional
        List of length P containing the names of the predictors.
        Ignored if ``data`` is provided (use ``X`` to specify columns).
        Default = None.
    add_intercept : :obj:`bool`, optional
        If True, an intercept column is automatically added to the predictor matrix.
        If False, the predictors matrix is passed as-is to estimators.
        Default = True.
    """

    def __init__(
        self, y=None, v=None, X=None, n=None, data=None, X_names=None, add_intercept=True
    ):
        if y is None and data is None:
            raise ValueError(
                "If no y values are provided, a pandas DataFrame "
                "containing a 'y' column must be passed to the "
                "data argument."
            )

        if (X is None) and (not add_intercept):
            raise ValueError("If no X matrix is provided, add_intercept must be True!")

        # Extract columns from DataFrame
        if data is not None:
            y = data.loc[:, y or "y"].values

            # v is optional
            if (v is not None) or ("v" in data.columns):
                v = data.loc[:, v or "v"].values

            # X is optional
            if (X is not None) or ("X" in data.columns):
                X_names = X or "X"
                X = data.loc[:, X_names].values

            # n is optional
            if (n is not None) or ("n" in data.columns):
                n = data.loc[:, n or "n"].values

        self.y = ensure_2d(y)
        self.v = ensure_2d(v)
        self.n = ensure_2d(n)
        X, names = self._get_predictors(X, X_names, add_intercept)
        self.X = X
        self.X_names = names

        _check_inputs_shape(self.y, self.X, "y", "X", row=True)
        _check_inputs_shape(self.y, self.v, "y", "v", row=True, column=True)
        _check_inputs_shape(self.y, self.n, "y", "n", row=True, column=True)

    def _get_predictors(self, X, names, add_intercept):
        if X is None and not add_intercept:
            raise ValueError(
                "No fixed predictors found. If no X matrix is "
                "provided, add_intercept must be True!"
            )

        X = pd.DataFrame(X)
        if names is not None:
            X.columns = _listify(names)

        if add_intercept:
            intercept = pd.DataFrame({"intercept": np.ones(len(self.y))})
            X = pd.concat([intercept, X], axis=1)

        return X.values, X.columns.tolist()

    def to_df(self):
        """Convert the dataset to a pandas DataFrame.

        Returns
        -------
        :obj:`pandas.DataFrame`
            A DataFrame containing the y, v, X, and n values.
        """
        if self.y.shape[1] == 1:
            df = pd.DataFrame({"y": self.y[:, 0]})

            if self.v is not None:
                df["v"] = self.v[:, 0]

            if self.n is not None:
                df["n"] = self.n[:, 0]

            df[self.X_names] = self.X

        else:
            all_dfs = []
            for i_set in range(self.y.shape[1]):
                df = pd.DataFrame(
                    {
                        "set": np.full(self.y.shape[0], i_set),
                        "y": self.y[:, i_set],
                    }
                )

                if self.v is not None:
                    df["v"] = self.v[:, i_set]

                if self.n is not None:
                    df["n"] = self.n[:, i_set]

                # X is the same across sets
                df[self.X_names] = self.X

                all_dfs.append(df)

            df = pd.concat(all_dfs, axis=0)

        return df


def meta_regression(
    y=None,
    v=None,
    X=None,
    n=None,
    data=None,
    X_names=None,
    add_intercept=True,
    method="ML",
    ci_method="QP",
    alpha=0.05,
    **kwargs,
):
    """Fit the standard meta-regression/meta-analysis model to provided data.

    Parameters
    ----------
    y : None or :obj:`numpy.ndarray` of shape (K,) or :obj:`str`, optional
        1d array of study-level estimates with length K, or the name of the column in data
        containing the y values.
        Default = None.
    v : None or :obj:`numpy.ndarray` of shape (K,) or :obj:`str`, optional
        1d array of study-level variances with length K, or the name of the column in data
        containing v values.
        Default = None.
    X : None or :obj:`numpy.ndarray` of shape (K,[P]) or :obj:`list` of :obj:`str`, optional
        1d or 2d array containing study-level predictors (dimensions K x P),
        or a list of strings giving the names of the columns in data containing the X values.
        Default = None.
    n : None or :obj:`numpy.ndarray` of shape (K,) or :obj:`str`, optional
        1d array of study-level sample sizes (length K), or the name of the corresponding column
        in ``data``.
        Default = None.
    data : None or :obj:`pandas.DataFrame` or :obj:`~pymare.core.Dataset`, optional
        If a Dataset instance is passed, the y, v, X, n and associated arguments are ignored,
        and data is passed directly to the selected estimator.
        If a pandas DataFrame, y, v, X and/or n values are taken from the DF columns.
        By default, columns are expected to have the same names as arguments
        (e.g., the y values will be expected in the 'y' column).
        This can be modified by passing strings giving column names to any of the y, v, X, or n
        arguments.
    X_names : None or :obj:`list` of :obj:`str`, optional
        List of length P containing the names of the predictors.
        Ignored if ``data`` is provided (use ``X`` to specify columns).
        Default = None.
    add_intercept : :obj:`bool`, optional
        If True, an intercept column is automatically added to the predictor matrix.
        If False, the predictors matrix is passed as-is to estimators.
        Default = True.
    method : {"ML", "REML", "DL", "HE", "WLS", "FE", "Stan"}, optional
        Name of estimation method. Default = 'ML'.
        Supported estimators include:

            - 'ML': Maximum-likelihood estimator
            - 'REML': Restricted maximum-likelihood estimator
            - 'DL': DerSimonian-Laird estimator
            - 'HE': Hedges estimator
            - 'WLS' or 'FE': Weighted least squares (fixed effects only)
            - 'Stan': Full Bayesian MCMC estimation via Stan
    ci_method : {"QP"}, optional
        Estimation method to use when computing uncertainty estimates.
        Currently only 'QP' is supported. Default = 'QP'.
        Ignored if ``method == 'Stan'``.
    alpha : :obj:`float`, optional
        Desired alpha level (CIs will have 1 - alpha coverage). Default = 0.05.
    **kwargs
        Optional keyword arguments to pass onto the chosen estimator.

    Returns
    -------
    :obj:`~pymare.results.MetaRegressionResults` or \
        :obj:`~pymare.results.BayesianMetaRegressionResults`
        A MetaRegressionResults or BayesianMetaRegressionResults instance,
        depending on the specified method ('Stan' will return the latter; all
        other methods return the former).
    """
    # if data is None or not isinstance(data, Dataset):
    if data is None or not data.__class__.__name__ == "Dataset":
        data = Dataset(y, v, X, n, data, X_names, add_intercept)

    method = method.lower()

    if method in ["ml", "reml"]:
        if v is not None:
            est_cls = partial(VarianceBasedLikelihoodEstimator, method=method)
        elif n is not None:
            est_cls = partial(SampleSizeBasedLikelihoodEstimator, method=method)
        else:
            raise ValueError("If method is ML or REML, one of `v` or `n` must be passed!")
    else:
        est_cls = {
            "dl": DerSimonianLaird,
            "wls": WeightedLeastSquares,
            "fe": WeightedLeastSquares,
            "stan": StanMetaRegression,
            "he": Hedges,
        }[method]

    # Get estimates
    est = est_cls(**kwargs)
    est.fit_dataset(data)
    return est.summary()
