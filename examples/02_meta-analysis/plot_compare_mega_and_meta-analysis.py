# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
.. _meta3:

================================================
 Run mega- and meta-analysis
================================================

Steps:

1.  Make a toy dataset
1.  Run mega-analysis (linear mixed effects model with random intercepts for site)
1.  Group dataset by site and run OLS on each site separately to construct meta-analysis dataset
1.  Run meta-analysis with DerSimonian-Laird between-study variance estimator
"""
import numpy as np
import statsmodels.api as sm

###############################################################################
# Wrangle some example data
# -----------------------------------------------------------------------------
data = sm.datasets.anes96.load_pandas().data
data.head()

###############################################################################
dat = data["popul"]
bins = np.linspace(0, np.max(dat) + 1, 20)
digitized = np.digitize(dat, bins)
idx = {i: np.where(digitized == i)[0] for i in range(1, len(bins))}
idx = {k: v for k, v in idx.items() if v.size}

# Assign "site" based on grouped populations
data["site"] = 0
i = 0
letters = ["a", "b", "c", "d", "e", "f", "g", "h"]
for k, v in idx.items():
    data.loc[v, "site"] = letters[i]
    i += 1

data.head()

###############################################################################
# The mega-analysis
# -----------------------------------------------------------------------------
# Random intercepts model using site
#
# Do age and education predict TV news watching?
model = sm.MixedLM.from_formula("TVnews ~ age + educ", data, groups=data["site"])
fitted_model = model.fit()
print(fitted_model.summary())

###############################################################################
# Create the meta-analysis dataset
# -----------------------------------
import pandas as pd

data["intercept"] = 1
target_vars = ["educ", "age", "intercept"]

# calculate mean and variance for each variance of interest
meta_df = []
for site_name, site_df in data.groupby("site"):
    model = sm.OLS(site_df["TVnews"], site_df[target_vars])
    fitted_model = model.fit()

    # extract parameter estimates and errors as Series
    coefficients = fitted_model.params
    std_errors = fitted_model.bse

    # convert standard error to sampling variance
    sampling_variances = std_errors ** 2
    names = [n + "_var" for n in sampling_variances.index.tolist()]
    sampling_variances.index = names

    # combine Series and convert to DataFrame
    coefficients = coefficients.append(sampling_variances)
    coefficients["site"] = site_name
    coefficients["sample_size"] = site_df.shape[0]
    temp_df = pd.DataFrame(coefficients).T
    meta_df.append(temp_df)

# Combine across sites and convert objects to floats
meta_df = pd.concat(meta_df).reset_index(drop=True)
meta_df = meta_df.convert_dtypes()
print(meta_df.to_markdown())

from pymare import Dataset

###############################################################################
# The meta-analysis
# --------------------------------------------
# Are age and education significant predictors of TV news watching across the literature?
from pymare.estimators import DerSimonianLaird

metamodel = DerSimonianLaird()
dset = Dataset(
    y=meta_df[["age", "educ"]].values.astype(float),
    v=meta_df[["age_var", "educ_var"]].values.astype(float),
    add_intercept=True,
)
metamodel.fit_dataset(dset)

###############################################################################
summary = metamodel.summary()
summary.get_fe_stats()
