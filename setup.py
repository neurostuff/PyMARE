#!/usr/bin/env python
"""PyMARE setup script."""
from setuptools import setup

import versioneer

if __name__ == "__main__":
    setup(
        name="PyMARE",
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        zip_safe=False,
    )
