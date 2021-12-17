# PyMARE: Python Meta-Analysis & Regression Engine
A Python library for mixed-effects meta-regression (including meta-analysis).

[![Latest Version](https://img.shields.io/pypi/v/pymare.svg)](https://pypi.python.org/pypi/pymare/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pymare.svg)](https://pypi.python.org/pypi/pymare/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/228903736.svg)](https://zenodo.org/badge/latestdoi/228903736)
[![Documentation Status](https://readthedocs.org/projects/pymare/badge/?version=latest)](http://pymare.readthedocs.io/en/latest/?badge=latest)
![GitHub CI](https://github.com/neurostuff/pymare/actions/workflows/testing.yml/badge.svg)
[![Codecov](https://codecov.io/gh/neurostuff/PyMARE/branch/master/graph/badge.svg)](https://codecov.io/gh/neurostuff/pymare)

**PyMARE is alpha software under heavy development; we reserve the right to make major changes to the API.**

## Quickstart
Install PyMARE from PyPI:
```
pip install pymare
```


Or for the bleeding-edge GitHub version:

```
pip install git+https://github.com/neurostuff/pymare.git
```

Suppose we have parameter estimates from 8 studies, along with corresponding variances, and a single (fixed) covariate:

```python
y = np.array([-1, 0.5, 0.5, 0.5, 1, 1, 2, 10]) # study-level estimates
v = np.array([1, 1, 2.4, 0.5, 1, 1, 1.2, 1.5]) # study-level variances
X = np.array([1, 1, 2, 2, 4, 4, 2.8, 2.8]) # a fixed study-level covariate
```

We can conduct a mixed-effects meta-regression using restricted maximum-likelihood (ReML)estimation in PyMARE using the high-level `meta_regression` function:

```python
from pymare import meta_regression

result = meta_regression(y, v, X, names=['my_cov'], add_intercept=True,
                         method='REML')
print(result.to_df())
```

This produces the following output:

```
         name   estimate        se   z-score     p-val  ci_0.025   ci_0.975
0  intercept  -0.106579  2.993715 -0.035601  0.971600 -5.974153   5.760994
1     my_cov   0.769961  1.113344  0.691575  0.489204 -1.412153   2.952075
```

Alternatively, we can achieve the same outcome using PyMARE's object-oriented API (which the `meta_regression` function wraps):

```python

from pymare import Dataset
from pymare.estimators import VarianceBasedLikelihoodEstimator

# A handy container we can pass to any estimator
dataset = Dataset(y, v, X)
# Estimator class for likelihood-based methods when variances are known
estimator = VarianceBasedLikelihoodEstimator(method='REML')
# All estimators expose a fit_dataset() method that takes a `Dataset`
# instance as the first (and usually only) argument.
estimator.fit_dataset(dataset)
# Post-fitting we can obtain a MetaRegressionResults instance via .summary()
results = estimator.summary()
# Print summary of results as a pandas DataFrame
print(result.to_df())
```

And if we want to be even more explicit, we can avoid the `Dataset` abstraction
entirely (though we'll lose some convenient validation checks):

```python
estimator = VarianceBasedLikelihoodEstimator(method='REML')

# X must be 2-d; this is one of the things the Dataset implicitly handles.
X = X[:, None]

estimator.fit(y, v, X)

results = estimator.summary()
```
