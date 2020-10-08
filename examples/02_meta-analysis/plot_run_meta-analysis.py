# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
.. _meta1:

========================
 Running a meta-analysis
========================

PyMARE implements a range of meta-analytic estimators.
"""
###############################################################################
# Start with the necessary imports
# --------------------------------
import numpy as np

from pymare import core, estimators

###############################################################################
# Here we simulate a dataset
# -----------------------------------
n_studies = 100
v = np.random.random((n_studies))
y = np.random.random((n_studies))
X = np.random.random((n_studies, 5))
n = np.random.randint(5, 50, size=n_studies)

###############################################################################
# Datasets can also be created from pandas DataFrames
# ---------------------------------------------------
dataset = core.Dataset(v=v, X=X, y=y, n=n)
est = estimators.WeightedLeastSquares().fit_dataset(dataset)
results = est.summary()
print(results.to_df())
