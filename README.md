# PyMARE: Python Meta-Analysis & Regression Engine
A Python library for mixed-effects meta-regression (including meta-analysis).

**PyMARE should be considered pre-alpha software, and the API is under active development. Expect things to break.**

## Quickstart
Install PyMARE from GitHub:

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
2      tau^2  10.949937       NaN       NaN       NaN  3.807599  59.616025
```

Alternatively, we can achieve the same outcome using PyMARE's object-oriented API (which the `meta_regression` function wraps):

```python

from pymare import Dataset
from pymare.estimators import LikelihoodBased

# A handy container we can pass to any estimator
dataset = Dataset(y, v, X)
# Estimator class for all likelihood-based methods
estimator = LikelihoodBased(method='REML')
# All estimators accept a `Dataset` instance as the first argument to `.fit()`
result = estimator.fit(dataset)
# Print summary of results as a pandas DataFrame
print(result.to_df())
```
