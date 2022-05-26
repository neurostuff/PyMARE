# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
.. _meta_basics:

=======================
Running a meta-analysis
=======================

PyMARE implements a range of meta-analytic estimators.
"""
###############################################################################
# Start with the necessary imports
# -----------------------------------------------------------------------------
from pymare import core, datasets, estimators

###############################################################################
# Here we load a dataset
# -----------------------------------------------------------------------------
# We will use the Michael 2013 dataset.
#
# We only want ot do a mean analysis, so we won't have any covariates except for
# an intercept.
data, meta = datasets.michael2013()
data["total_sample_size"] = data["No_brain_n"] + data["Brain_n"]
dset = core.Dataset(data=data, y="yi", v="vi", n="total_sample_size", X=None, add_intercept=True)

###############################################################################
# Now we fit a model
# -----------------------------------------------------------------------------
est = estimators.WeightedLeastSquares().fit_dataset(dset)
results = est.summary()
print(results.to_df())
