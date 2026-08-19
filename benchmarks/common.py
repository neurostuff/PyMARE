"""Shared data builders for the benchmark suite.

Every benchmark draws its data from the same seeded generator, so a timing
difference between two commits is a difference in the code and never in the
input.
"""

import numpy as np

#: Observations per dataset, datasets analyzed side by side, and observations
#: per group. Sized so the vectorized paths dominate but the suite still runs in
#: seconds -- see the module docstring in ``__init__.py``.
N_OBSERVATIONS = 200
N_DATASETS = 500
OBSERVATIONS_PER_GROUP = 4

#: Estimators that loop over datasets rather than vectorizing across them get a
#: smaller second dimension, or they would dominate the whole suite.
N_DATASETS_LOOPED = 10


def make_data(
    n_observations=N_OBSERVATIONS,
    n_datasets=N_DATASETS,
    n_predictors=1,
    observations_per_group=OBSERVATIONS_PER_GROUP,
    seed=0,
):
    """Build a meta-regression problem with dependent observations.

    Parameters
    ----------
    n_observations : :obj:`int`, optional
        Rows of ``y`` and ``v``.
    n_datasets : :obj:`int`, optional
        Columns of ``y`` and ``v``: independent problems sharing one design.
    n_predictors : :obj:`int`, optional
        Predictors beyond the intercept.
    observations_per_group : :obj:`int`, optional
        Group size. Groups are contiguous and equally sized.
    seed : :obj:`int`, optional
        Seed for the generator.

    Returns
    -------
    y, v, X, groups : :obj:`numpy.ndarray`
        Effects, sampling variances, design and group labels.
    """
    rng = np.random.RandomState(seed)
    y = rng.standard_normal((n_observations, n_datasets))
    v = np.abs(rng.standard_normal((n_observations, n_datasets))) + 0.5
    X = np.c_[
        np.ones(n_observations),
        rng.standard_normal((n_observations, n_predictors - 1)),
    ]
    groups = np.repeat(np.arange(n_observations // observations_per_group), observations_per_group)
    return y, v, X, groups


def make_sample_sizes(n_observations=N_OBSERVATIONS, n_datasets=N_DATASETS, seed=0):
    """Build sample sizes for the estimators that take ``n`` instead of ``v``."""
    rng = np.random.RandomState(seed)
    return rng.randint(20, 200, size=(n_observations, n_datasets)).astype(float)


def make_group_level_design(groups, n_predictors=3, seed=1):
    """Build a design whose predictors are constant within each group.

    The aggregating weight schemes reduce every group to one row, so they only
    accept a design that does not vary inside a group. Only DerSimonian-Laird
    reads the observation-level design, and only under ``"rescale"``.

    Parameters
    ----------
    groups : :obj:`numpy.ndarray`
        Group labels, one per observation.
    n_predictors : :obj:`int`, optional
        Predictors including the intercept.
    seed : :obj:`int`, optional
        Seed for the generator.

    Returns
    -------
    :obj:`numpy.ndarray`
        Design of shape ``(groups.size, n_predictors)``.
    """
    rng = np.random.RandomState(seed)
    labels, inverse = np.unique(groups, return_inverse=True)
    per_group = rng.standard_normal((labels.size, n_predictors - 1))
    return np.c_[np.ones(groups.size), per_group[inverse]]
