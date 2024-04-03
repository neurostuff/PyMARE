"""PyMARE: Python Meta-Analysis & Regression Engine."""

import sys
import warnings

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


def _py367_deprecation_warning():
    """Deprecation warnings message.

    Notes
    -----
    Adapted from NiMARE.
    """
    py36_warning = (
        "Python 3.6 and 3.7 support is deprecated and will be removed in release 0.0.5 of PyMARE. "
        "Consider switching to Python 3.8, 3.9."
    )
    warnings.filterwarnings("once", message=py36_warning)
    warnings.warn(message=py36_warning, category=FutureWarning, stacklevel=3)


def _python_deprecation_warnings():
    """Raise deprecation warnings.

    Notes
    -----
    Adapted from NiMARE.
    """
    if sys.version_info.major == 3 and (
        sys.version_info.minor == 6 or sys.version_info.minor == 7
    ):
        _py367_deprecation_warning()


_python_deprecation_warnings()
