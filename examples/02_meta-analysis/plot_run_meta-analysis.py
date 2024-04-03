# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
.. _meta_basics:

=====================================
The Basics of Running a Meta-Analysis
=====================================

Here we walk through the basic steps of running a meta-analysis with PyMARE.
"""
###############################################################################
# Start with the necessary imports
# -----------------------------------------------------------------------------
from pprint import pprint

from pymare import core, datasets, estimators

###############################################################################
# Load the data
# -----------------------------------------------------------------------------
# We will use the :footcite:t:`michael2013non` dataset, which comes from the
# metadat library :footcite:p:`white2022metadat`.
#
# We only want to do a mean analysis, so we won't have any covariates except for
# an intercept.
data, meta = datasets.michael2013()
dset = core.Dataset(data=data, y="yi", v="vi", X=None, add_intercept=True)
dset.to_df()

###############################################################################
# Now we fit a model
# -----------------------------------------------------------------------------
# You must first initialize the estimator, after which you can use
# :meth:`~pymare.estimators.estimators.BaseEstimator.fit` to fit the model to
# numpy arrays, or
# :meth:`~pymare.estimators.estimators.BaseEstimator.fit_dataset` to fit it to
# a :class:`~pymare.core.Dataset`.
#
# .. tip::
#   We generally recommend using
#   :meth:`~pymare.estimators.estimators.BaseEstimator.fit_dataset` over
#   :meth:`~pymare.estimators.estimators.BaseEstimator.fit`.
#
#   There are a number of methods, such as
#   :meth:`~pymare.results.MetaRegressionResults.get_heterogeneity_stats` and
#   :meth:`~pymare.results.MetaRegressionResults.permutation_test`,
#   which only work when the Estimator is fitted to a Dataset.
#
#   However, :meth:`~pymare.estimators.estimators.BaseEstimator.fit` requires
#   less memory than :meth:`~pymare.estimators.estimators.BaseEstimator.fit_dataset`,
#   so it can be useful for large-scale meta-analyses,
#   such as neuroimaging image-based meta-analyses.
#
# The :meth:`~pymare.estimators.estimators.BaseEstimator.summary` function
# will return a :class:`~pymare.results.MetaRegressionResults` object,
# which contains the results of the analysis.
est = estimators.WeightedLeastSquares().fit_dataset(dset)
results = est.summary()
results.to_df()

###############################################################################
# We can also extract some useful information from the results object
# -----------------------------------------------------------------------------
# The :meth:`~pymare.results.MetaRegressionResults.get_heterogeneity_stats`
# method will calculate heterogeneity statistics.
pprint(results.get_heterogeneity_stats())

###############################################################################
# The :meth:`~pymare.results.MetaRegressionResults.get_re_stats` method will
# estimate the confidence interval for :math:`\tau^2`.
pprint(results.get_re_stats())

###############################################################################
# The :meth:`~pymare.results.MetaRegressionResults.permutation_test` method
# will run a permutation test to estimate more accurate p-values.
perm_results = results.permutation_test(n_perm=1000)
perm_results.to_df()

###############################################################################
# References
# -----------------------------------------------------------------------------
# .. footbibliography::
