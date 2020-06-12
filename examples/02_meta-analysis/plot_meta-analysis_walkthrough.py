# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
.. _meta1:

================================================
 Run a set of Estimators on a simulated dataset
================================================

PyMARE implements a range of meta-analytic estimators.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

import pymare


# A small function to make things easier later on
def var_to_ci(y, v, n):
    """Convert sampling variance to 95% CI"""
    term = 1.96 * np.sqrt(v) / np.sqrt(n)
    return y - term, y + term


###############################################################################
# Here we simulate a dataset
# -----------------------------------
n_studies = 40
between_study_var = 400  # population variance
between_study_sd = np.sqrt(between_study_var)
true_effect = 20
sample_sizes = np.round(np.random.normal(loc=50, scale=20, size=n_studies)).astype(int)
within_study_vars = np.random.normal(loc=400, scale=100, size=n_studies)
study_means = np.random.normal(loc=true_effect, scale=between_study_sd, size=n_studies)

sample_sizes[sample_sizes <= 1] = 2
within_study_vars = np.abs(within_study_vars)

# Convert data types and match PyMARE nomenclature
y = study_means
X = np.ones((n_studies))
v = within_study_vars
n = sample_sizes
sd = np.sqrt(v * n)
z = y / sd
p = stats.norm.sf(abs(z)) * 2

###############################################################################
# Plot variable distributions
# -----------------------------------
fig, axes = plt.subplots(nrows=5, figsize=(16, 10))
sns.distplot(y, ax=axes[0], bins=20)
axes[0].set_title('y')
sns.distplot(v, ax=axes[1], bins=20)
axes[1].set_title('v')
sns.distplot(n, ax=axes[2], bins=20)
axes[2].set_title('n')
sns.distplot(z, ax=axes[3], bins=20)
axes[3].set_title('z')
sns.distplot(p, ax=axes[4], bins=20)
axes[4].set_title('p')
for i in range(5):
    axes[i].set_yticks([])
fig.tight_layout()
fig.show()

###############################################################################
# Plot means and confidence intervals
# -----------------------------------
# Here we can show study-wise mean effect and CIs, along with the true effect
# and CI corresponding to the between-study variance.
fig, ax = plt.subplots(figsize=(8, 16))
study_ticks = np.arange(n_studies)

# Get 95% CI for individual studies
lower_bounds, upper_bounds = var_to_ci(y, v, n)
ax.scatter(y, study_ticks+1)
for study in study_ticks:
    ax.plot((lower_bounds[study], upper_bounds[study]), (study+1, study+1), color='blue')
ax.axvline(0, color='gray', alpha=0.2, linestyle='--', label='Zero')
ax.axvline(np.mean(y), color='orange', alpha=0.2, label='Mean of Observed Effects')

# Get 95% CI for true effect
lower_bound, upper_bound = var_to_ci(true_effect, between_study_var, 1)
ax.scatter((true_effect,), (n_studies+1,), color='green', label='True Effect')
ax.plot((lower_bound, upper_bound), (n_studies+1, n_studies+1),
        color='green', linewidth=3, label='Between-Study 95% CI')
ax.set_ylim((0, n_studies+2))
ax.set_xlabel('Mean (95% CI)')
ax.set_ylabel('Study')
ax.legend()
fig.show()

###############################################################################
# Create a Dataset object containing the data
# --------------------------------------------
dset = pymare.Dataset(y=y, X=None, v=v, n=n, add_intercept=True)

# Here is a dictionary to house results across models
results = {}

###############################################################################
# Fit models
# -----------------------------------------------------------------------------
# When you have ``z`` or ``p``:
#
# - :class:`pymare.estimators.Stouffers`
# - :class:`pymare.estimators.Fishers`
#
# When you have ``y`` and ``v`` and don't want to estimate between-study variance:
#
# - :class:`pymare.estimators.WeightedLeastSquares`
#
# When you have ``y`` and ``v`` and want to estimate between-study variance:
#
# - :class:`pymare.estimators.DerSimonianLaird`
# - :class:`pymare.estimators.Hedges`
# - :class:`pymare.estimators.VarianceBasedLikelihoodEstimator`
#
# When you have ``y`` and ``n`` and want to estimate between-study variance:
#
# - :class:`pymare.estimators.SampleSizeBasedLikelihoodEstimator`
#
# When you have ``y`` and ``v`` and want a hierarchical model:
#
# - :class:`pymare.estimators.StanMetaRegression`

###############################################################################
# First, we have "combination models", which combine p and/or z values
# `````````````````````````````````````````````````````````````````````````````
# The two combination models in PyMARE are Stouffer's and Fisher's Tests.
#
# Notice that these models don't use :class:`pymare.core.Dataset` objects.
stouff = pymare.estimators.Stouffers(input='z')
stouff.fit(y=z[:, None])
stouff_summary = stouff.summary()
print('z: {}'.format(stouff_summary.z))
print('p: {}'.format(stouff_summary.p))

fisher = pymare.estimators.Stouffers(input='z')
fisher.fit(y=z[:, None])
fisher_summary = fisher.summary()
print('z: {}'.format(fisher_summary.z))
print('p: {}'.format(fisher_summary.p))

###############################################################################
# Now we have a fixed effects model
# `````````````````````````````````````````````````````````````````````````````
# This estimator does not attempt to estimate between-study variance.
# Instead, it takes `tau^2` as an argument.
wls = pymare.estimators.WeightedLeastSquares()
wls.fit(dset)
wls_summary = wls.summary()
results['Weighted Least Squares'] = wls_summary.to_df()
print(wls_summary.to_df())

###############################################################################
# Methods that estimate between-study variance
# `````````````````````````````````````````````````````````````````````````````
# The `DerSimonianLaird`, `Hedges`, and `VarianceBasedLikelihoodEstimator`
# estimators all estimate between-study variance from the data, and use ``y``
# and ``v``.
#
# `DerSimonianLaird` and `Hedges` use relatively simple methods for estimating
# between-study variance, while `VarianceBasedLikelihoodEstimator` can use
# either maximum-likelihood (ML) or restricted maximum-likelihood (REML) to
# iteratively estimate it.
dsl = pymare.estimators.DerSimonianLaird()
dsl.fit(dset)
dsl_summary = dsl.summary()
results['DerSimonian-Laird'] = dsl_summary.to_df()
print(dsl_summary.to_df())

hedge = pymare.estimators.Hedges()
hedge.fit(dset)
hedge_summary = hedge.summary()
results['Hedges'] = hedge_summary.to_df()
print(hedge_summary.to_df())

vb_ml = pymare.estimators.VarianceBasedLikelihoodEstimator(method='ML')
vb_ml.fit(dset)
vb_ml_summary = vb_ml.summary()
results['Variance-Based with ML'] = vb_ml_summary.to_df()
print(vb_ml_summary.to_df())

vb_reml = pymare.estimators.VarianceBasedLikelihoodEstimator(method='REML')
vb_reml.fit(dset)
vb_reml_summary = vb_reml.summary()
results['Variance-Based with REML'] = vb_reml_summary.to_df()
print(vb_reml_summary.to_df())

# The `SampleSizeBasedLikelihoodEstimator` estimates between-study variance
# using ``y`` and ``n``, but assumes within-study variance is homogenous
# across studies.
sb_ml = pymare.estimators.SampleSizeBasedLikelihoodEstimator(method='ML')
sb_ml.fit(dset)
sb_ml_summary = sb_ml.summary()
results['Sample Size-Based with ML'] = sb_ml_summary.to_df()
print(sb_ml_summary.to_df())

sb_reml = pymare.estimators.SampleSizeBasedLikelihoodEstimator(method='REML')
sb_reml.fit(dset)
sb_reml_summary = sb_reml.summary()
results['Sample Size-Based with REML'] = sb_reml_summary.to_df()
print(sb_reml_summary.to_df())

###############################################################################
# What about the Stan estimator?
# `````````````````````````````````````````````````````````````````````````````
# We're going to skip this one here because of how computationally intensive it
# is.

###############################################################################
# Let's check out our results!
# `````````````````````````````````````````````````````````````````````````````
fig, ax = plt.subplots(figsize=(8, 8))

for i, (estimator_name, summary_df) in enumerate(results.items()):
    ax.scatter((summary_df.loc[0, 'estimate'],), (i+1,), label=estimator_name)
    ax.plot((summary_df.loc[0, 'ci_0.025'],  summary_df.loc[0, 'ci_0.975']),
            (i+1, i+1),
            linewidth=3)

# Get 95% CI for true effect
lower_bound, upper_bound = var_to_ci(true_effect, between_study_var, 1)
ax.scatter((true_effect,), (i+2,), label='True Effect')
ax.plot((lower_bound, upper_bound), (i+2, i+2),
        linewidth=3, label='Between-Study 95% CI')
ax.set_ylim((0, i+3))
ax.set_yticklabels([None] + list(results.keys()) + ['True Effect'])

ax.set_xlabel('Mean (95% CI)')
fig.show()
