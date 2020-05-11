# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
.. _io1:

===================
 Creating a dataset
===================

In PyMARE, operations are performed on `Dataset` objects.
Here we show how Datasets are loaded and what they can be used for.

.. note::
    This will likely change as we work to shift database querying to a remote
    database, rather than handling it locally with NiMARE.
"""
###############################################################################
# Start with the necessary imports
# --------------------------------
import numpy as np
import pandas as pd

from pymare.core import Dataset

###############################################################################
# Datasets can be created from arrays
# -----------------------------------
v = [100, 100, 100]
X = [[5, 2, 1], [9, 8, 7]]
y = [2, 4, 6]
dataset = Dataset(y=y, v=v, X=X)

###############################################################################
# Datasets can also be created from pandas DataFrames
# ---------------------------------------------------
df = pd.DataFrame({
    'y': [2, 4, 6],
    'v_alt': [100, 100, 100],
    'X1': [5, 2, 1],
    'X7': [9, 8, 7]
})
dataset = Dataset(v='v_alt', X=['X1', 'X7'], data=df)
