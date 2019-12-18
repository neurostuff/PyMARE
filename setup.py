#! /usr/bin/env python

from setuptools import setup, find_packages


AUTHOR = 'PyMARE developers'
COPYRIGHT = 'Copyright 2019--now, PyMARE developers'
URL = 'https://github.com/neurostuff/PyMARE'
DISTNAME = 'PyMARE'
DESCRIPTION = 'Python Meta-Analysis & Regression Engine'
MAINTAINER = 'Tal Yarkoni'
MAINTAINER_EMAIL = 'tyarkoni@gmail.com'
LICENSE = 'MIT'
VERSION = '0.0.1'
INSTALL_REQUIRES = ['numpy', 'scipy', 'pandas']


if __name__ == "__main__":
    setup(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        author=AUTHOR,
        license=LICENSE,
        version=VERSION,
        zip_safe=False,
        packages=find_packages(),
        package_data={},
        install_requires=INSTALL_REQUIRES,
        python_requires='>=3.5',
    )
