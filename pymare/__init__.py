from .core import Dataset, meta_regression
from .effectsize import (OneSampleEffectSizeConverter,
                          TwoSampleEffectSizeConverter)

from ._version import get_versions

__version__ = get_versions()['version']

__all__ = [
    'Dataset',
    'meta_regression',
    'OneSampleEffectSizeConverter',
    'TwoSampleEffectSizeConverter'
]

del get_versions
