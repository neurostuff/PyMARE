"""
.. _io1:

==================
Creating a dataset
==================

In PyMARE, operations are performed on :class:`~pymare.core.Dataset` objects.
Datasets are very lightweight objects that store the data used for
meta-analyses, including study-level estimates (y), variances (v),
predictors (X), and sample sizes (n).
"""
###############################################################################
# Start with the necessary imports
# --------------------------------
from pprint import pprint

import pandas as pd

from pymare import core

###############################################################################
# Datasets can be created from arrays
# -----------------------------------
# The simplest way to create a dataset is to pass in arguments as numpy arrays.
#
# ``y`` refers to the study-level estimates, ``v`` to the variances,
# ``X`` to any study-level regressors, and ``n`` to the sample sizes.
#
# Not all Estimators require all of these arguments, so not all need to be
# used in a given Dataset.
y = [2, 4, 6]
v = [100, 100, 100]
X = [[5, 9], [2, 8], [1, 7]]

dataset = core.Dataset(y=y, v=v, X=X, X_names=["X1", "X7"])

pprint(vars(dataset))

###############################################################################
# Datasets have the :meth:`~pymare.core.Dataset.to_df` method.
dataset.to_df()

###############################################################################
# Datasets can also be created from pandas DataFrames
# ---------------------------------------------------
df = pd.DataFrame(
    {
        "y": [2, 4, 6],
        "v_alt": [100, 100, 100],
        "X1": [5, 2, 1],
        "X7": [9, 8, 7],
    }
)

dataset = core.Dataset(v="v_alt", X=["X1", "X7"], data=df, add_intercept=False)

pprint(vars(dataset))

###############################################################################
# Datasets can also contain multiple dependent variables
# ------------------------------------------------------
# These variables are analyzed in parallel, but as unrelated variables,
# rather than as potentially correlated ones.
#
# This is particularly useful for image-based neuroimaging meta-analyses.
# For more information about this, see `NiMARE <https://nimare.readthedocs.io>`_.
y = [
    [2, 4, 6],  # Estimates for first study's three outcome variables.
    [3, 2, 1],  # Estimates for second study's three outcome variables.
]
v = [
    [100, 100, 100],  # Estimate variances for first study's three outcome variables.
    [8, 4, 2],  # Estimate variances for second study's three outcome variables.
]
X = [
    [5, 9],  # Predictors for first study. Same across all three outcome variables.
    [2, 8],  # Predictors for second study. Same across all three outcome variables.
]

dataset = core.Dataset(y=y, v=v, X=X, X_names=["X1", "X7"])

pprint(vars(dataset))
