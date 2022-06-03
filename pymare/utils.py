"""Miscellaneous utility functions."""
import os.path as op

import numpy as np


def get_resource_path():
    """Return the path to general resources, terminated with separator.

    Resources are kept outside package folder in "datasets".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    return op.abspath(op.join(op.dirname(__file__), "resources") + op.sep)


def _listify(obj):
    """Wrap all non-list or tuple objects in a list.

    This provides a simple way to accept flexible arguments.
    """
    return obj if isinstance(obj, (list, tuple, type(None), np.ndarray)) else [obj]


def _get_sig_code(p_value):
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    elif p_value < 0.1:
        return "."
    else:
        return " "
