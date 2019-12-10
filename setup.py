#! /usr/bin/env python

from setuptools import setup, find_packages


DISTNAME = 'pymares'
DESCRIPTION = 'Python Meta-Analysis and Regression Engine'
MAINTAINER = 'Tal Yarkoni'
MAINTAINER_EMAIL = 'tyarkoni@gmail.com'
LICENSE = 'MIT'
VERSION = '0.0.1'
INSTALL_REQUIRES = ['numpy', 'scipy']


if __name__ == "__main__":
    setup(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        version=VERSION,
        zip_safe=False,
        packages=find_packages(),
        package_data={},
        install_requires=INSTALL_REQUIRES,
        python_requires='>=3.5',
    )
