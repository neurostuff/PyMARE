# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
.. _meta3:

================================================
 Run mega- and meta-analysis
================================================

Steps:

1.  Make a toy dataset
2.  Run mega-analysis (linear mixed effects model with random intercepts for study)
3.  Group dataset by study and run OLS on each study separately to construct meta-analysis dataset
4.  Run meta-analysis with DerSimonian-Laird between-study variance estimator
"""
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from pymare import Dataset
from pymare.estimators import DerSimonianLaird

sns.set_theme(style="ticks")

###############################################################################
# Wrangle some example data
# -----------------------------------------------------------------------------
# For this example, we will use the "American National Election Survey 1996"
# dataset from statsmodels.
# This dataset includes several elements we care about, including:
#
# 1. A large number of observations (944).
# 2. A variable that can be used as a substitute for "study".
#    In this case, we will pretend that all observations with similar
#    populations come from the same cities, which we will in turn pretend
#    reflect different studies of the same phenomenon by different research
#    groups.
data = sm.datasets.anes96.load_pandas().data
data.head(10)

###############################################################################
# Convert the populations to "studies" by binning similar populations and
# assigning them to the same "study" values.

# I selected the first study with a first author starting with each letter from Neurosynth.
# I figured it would give the dataset some verisimilitude.
study_names = [
    "Aarts & Roelofs (2011)",
    "Baas et al. (2008)",
    "Cabanis et al. (2013)",
    "D'Agata et al. (2011)",
    "Eack et al. (2008)",
    "Fabbri, Caramazza, & Lingnau (2012)",
    "Gaab, Gabrieli, & Glover (2007)",
    "Haag et al. (2014)",
]

data = data.sort_values(by="popul")
group_assignments = []
split_assignments = np.array_split(np.arange(data.shape[0], dtype=int), 8)
for i_split, split in enumerate(split_assignments):
    group_assignments += [study_names[i_split]] * len(split)

data["study"] = group_assignments
data.head(10)

###############################################################################
fig, ax = plt.subplots(figsize=(16, 8))
sns.stripplot(data=data, x="logpopul", y="study", ax=ax)
fig.tight_layout()
fig.show()

###############################################################################
# The variables of interest
# -----------------------------------------------------------------------------
# First, let's talk about the variables of interest in this analysis.
# The variables are:
#
# 1. TVnews: The number of times, per week, that the respondent watches the news on TV.
# 2. age: The age of the respondent, in years.
# 3. educ: The education of the respondent, binned into the following groups:
#    (1) Grade 1-8, (2) Some high school, (3) High school graduate, (4) Some college,
#    (5) College graduate, (6) Master's degree, and (7) PhD.
#    We will evaluate linear relationships with this coded variable, which isn't
#    optimal, but this is just an example, so... ¯\\_(ツ)_/¯
#
# If we visualize the distributions of the different variables, we'll see that
# they're not exactly normally distributed...
sns.pairplot(data[["study", "TVnews", "age", "educ"]], hue="study")

###############################################################################
# The mega-analysis
# -----------------------------------------------------------------------------
# Random intercepts model using study.
# In this model, we test the question, "to what extent do age and education
# predict TV news watching?"
#
# We assume that data from the same study may have different baseline news-watching levels
# (i.e., random intercepts).
#
# Do age and education predict TV news watching?
model = sm.MixedLM.from_formula("TVnews ~ age + educ", data, groups=data["study"])
fitted_model = model.fit()
print(fitted_model.summary())

###############################################################################
# Create the meta-analysis dataset
# -----------------------------------
# We assume that each study performed their own analyses testing the same hypothesis.
# The hypothesis, in this case, is "to what extent do age and education predict TV news watching"?
#
# The model used will, for our example, be exactly the same across studies.
# This model is just a GLM with age, education, and an intercept predicting the number of hours.
# Individual studies would then report the parameter estimate and variance for each term in the
# model.
data["intercept"] = 1
target_vars = ["educ", "age", "intercept"]

# calculate coefficient and variance for each variable of interest
meta_df = []
for study_name, study_df in data.groupby("study"):
    model = sm.OLS(study_df["TVnews"], study_df[target_vars])
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
    coefficients["study"] = study_name
    coefficients["sample_size"] = study_df.shape[0]
    temp_df = pd.DataFrame(coefficients).T
    meta_df.append(temp_df)

# Combine across studies and convert objects to floats
meta_df = pd.concat(meta_df).reset_index(drop=True)
meta_df = meta_df.convert_dtypes()
meta_df

###############################################################################
# The meta-analysis
# --------------------------------------------
# Are age and education significant predictors of TV news watching across the literature?
metamodel = DerSimonianLaird()
dset = Dataset(
    y=meta_df[["age", "educ"]].values.astype(float),
    v=meta_df[["age_var", "educ_var"]].values.astype(float),
    add_intercept=True,
)
metamodel.fit_dataset(dset)

###############################################################################
summary = metamodel.summary()
results = summary.get_fe_stats()
pprint(results)
