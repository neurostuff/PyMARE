.. include:: links.rst

Installation
============

PyMARE can be installed from pip. To install the latest official release:

.. code-block:: bash

    pip install pymare

If you want to use the most up-to-date version, you can install from the ``master`` branch:

.. code-block:: bash

    pip install git+https://github.com/neurostuff/PyMARE.git

PyMARE requires Python >=3.9 and a number of packages.
For a complete list, please see ``setup.cfg``.

Bayesian estimation with Stan
-----------------------------

:class:`~pymare.estimators.StanMetaRegression` is optional, and needs two
installation steps rather than one:

.. code-block:: bash

    pip install pymare[stan]
    python -m cmdstanpy.install_cmdstan

The first installs CmdStanPy. The second fetches and builds CmdStan itself,
which is a C++ program rather than a Python package and so needs a C++ toolchain
(``g++`` and ``make`` on Linux, the Command Line Tools on macOS, RTools on
Windows). It takes several minutes, once per machine.

The Stan model is compiled the first time the estimator is fitted, which takes
roughly another minute. The compiled model is cached alongside the installed
package, so later fits and later processes reuse it. If PyMARE is installed
somewhere unwritable, the model is compiled into ``~/.pymare/stan`` instead and a
warning says so.

Every other estimator in PyMARE is pure Python and needs none of this.
