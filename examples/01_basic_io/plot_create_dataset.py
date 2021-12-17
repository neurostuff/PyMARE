# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
.. _io1:

===================
 Creating a dataset
===================

In PyMARE, operations are performed on :class:`pymare.core.Dataset` objects.
Datasets are very lightweight objects that store the data used for
meta-analyses, including study-level estimates (y), variances (v),
predictors (X), and sample sizes (n).
"""
###############################################################################
# Start with the necessary imports
# --------------------------------
import pandas as pd
from pprint import pprint

from pymare import core

###############################################################################
# Datasets can be created from arrays
# -----------------------------------
v = [100, 100, 100]
X = [[5, 9], [2, 8], [1, 7]]
y = [2, 4, 6]
dataset = core.Dataset(y=y, v=v, X=X)

###############################################################################
# Datasets can also be created from pandas DataFrames
# ---------------------------------------------------
df = pd.DataFrame({"y": [2, 4, 6], "v_alt": [100, 100, 100], "X1": [5, 2, 1], "X7": [9, 8, 7]})
dataset = core.Dataset(v="v_alt", X=["X1", "X7"], data=df, add_intercept=False)

pprint(vars(dataset))
