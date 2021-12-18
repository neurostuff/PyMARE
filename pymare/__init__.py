"""PyMARE: Python Meta-Analysis & Regression Engine."""
from .core import Dataset, meta_regression
from .effectsize import OneSampleEffectSizeConverter, TwoSampleEffectSizeConverter

__all__ = [
    "Dataset",
    "meta_regression",
    "OneSampleEffectSizeConverter",
    "TwoSampleEffectSizeConverter",
]

from . import _version

__version__ = _version.get_versions()["version"]
del _version
