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


def _check_inputs_shape(param1, param2, param1_name, param2_name, row=False, column=False):
    """Check whether 'param1' and 'param2' have the same shape.

    Parameters
    ----------
    param1 : array
    param2 : array
    param1_name : str
    param2_name : str
    row : bool, default to False.
    column : bool, default to False.
    """
    if (param1 is not None) and (param2 is not None):
        if row and not column:
            shape1 = param1.shape[0]
            shape2 = param2.shape[0]
            message = "rows"
        elif column and not row:
            shape1 = param1.shape[1]
            shape2 = param2.shape[1]
            message = "columns"
        elif row and column:
            shape1 = param1.shape
            shape2 = param2.shape
            message = "rows and columns"
        else:
            raise ValueError("At least one of the two parameters (row or column) should be True.")

        if shape1 != shape2:
            raise ValueError(
                f"{param1_name} and {param2_name} should have the same number of {message}. "
                f"You provided {param1_name} with shape {param1.shape} and {param2_name} "
                f"with shape {param2.shape}."
            )
