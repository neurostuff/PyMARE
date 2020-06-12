import json
import os.path as op
import importlib.util

spec = importlib.util.spec_from_file_location(
    '_version', op.join(op.dirname(__file__), 'pymare/_version.py'))
_version = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_version)

VERSION = _version.get_versions()['version']
del _version

# Get list of authors from Zenodo file
with open(op.join(op.dirname(__file__), '.zenodo.json'), 'r') as fo:
    zenodo_info = json.load(fo)
authors = [author['name'] for author in zenodo_info['creators']]
authors = [author.split(', ')[1] + ' ' + author.split(', ')[0] for author in authors]

readme_file = op.join(op.dirname(__file__), 'README.md')
with open(readme_file, encoding='utf-8') as f:
    longdesc = f.read()

AUTHOR = 'PyMARE developers'
COPYRIGHT = 'Copyright 2019--now, PyMARE developers'
CREDITS = authors
LICENSE = 'MIT'
MAINTAINER = 'Tal Yarkoni'
EMAIL = 'tyarkoni@gmail.com'
STATUS = 'Prototype'
URL = 'https://github.com/neurostuff/PyMARE'
PACKAGENAME = 'PyMARE'
DESCRIPTION = 'PyMARE: Python Meta-Analysis & Regression Engine'
LONGDESC = longdesc
LONGDESCCONTTYPE = "text/markdown"

DOWNLOAD_URL = (
    'https://github.com/neurostuff/{name}/archive/{ver}.tar.gz'.format(
        name=PACKAGENAME, ver=VERSION))

REQUIRES = [
    'numpy>=1.8.0',
    'scipy',
    'pandas',
    'sympy',
    'wrapt'
]

STAN_REQUIRES = [
    'pystan',
    'arviz'
]

TESTS_REQUIRES = [
    'codecov',
    'coverage',
    'coveralls',
    'flake8',
    'pytest',
    'pytest-cov'
]

DOC_REQUIRES = [
    'sphinx~=2.4.2',
    'sphinx_rtd_theme',
    'sphinx-argparse',
    'numpydoc',
    'm2r',
    'sphinx_copybutton',
    'sphinx_gallery',
    'pillow',
    'matplotlib',
    'seaborn'
]

EXTRA_REQUIRES = {
    'stan': STAN_REQUIRES,
    'doc': DOC_REQUIRES,
    'tests': TESTS_REQUIRES,
}

# Enable a handle to install all extra dependencies at once
EXTRA_REQUIRES['all'] = list(set([
    v for deps in EXTRA_REQUIRES.values() for v in deps]))

ENTRY_POINTS = {}

# Package classifiers
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Topic :: Scientific/Engineering'
]
