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
est = estimators.WeightedLeastSquares().fit_dataset(dset)
results = est.summary()
results.to_df()

###############################################################################
# We can also extract some useful information from the results object
# -----------------------------------------------------------------------------
# Here we'll take a look at the heterogeneity statistics.
pprint(results.get_heterogeneity_stats())

###############################################################################
# We can also estimate the confidence interval for :math:`\tau^2`.
pprint(results.get_re_stats())

###############################################################################
# References
# -----------------------------------------------------------------------------
# .. footbibliography::
