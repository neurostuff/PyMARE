"""Tests for pymare.utils."""
import os.path as op

from pymare import utils


def test_get_resource_path():
    """Test nimare.utils.get_resource_path."""
    print(utils.get_resource_path())
    assert op.isdir(utils.get_resource_path())
