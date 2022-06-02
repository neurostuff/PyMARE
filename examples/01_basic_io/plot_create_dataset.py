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
y = [2, 4, 6]
v = [100, 100, 100]
X = [[5, 9], [2, 8], [1, 7]]

dataset = core.Dataset(y=y, v=v, X=X, X_names=["X1", "X7"])

pprint(vars(dataset))

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
# Datasets can also contain parallel "datasets"
# ---------------------------------------------
y = [
    [2, 4, 6],  # estimates for first "dataset"
    [3, 2, 1],  # estimates for second "dataset"
]
v = [
    [100, 100, 100],  # variance for first "dataset"'s estimates
    [8, 4, 2],  # variance for second "dataset"'s estimates
]
X = [  # all "dataset"s must have the same regressors
    [5, 9],
    [2, 8],
    [1, 7],
]

dataset = core.Dataset(y=y, v=v, X=X, X_names=["X1", "X7"])

pprint(vars(dataset))
