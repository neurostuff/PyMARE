API
===

.. _api_core_ref:

:mod:`pymare.core`: Core objects
--------------------------------------------------

.. automodule:: pymare.core
   :no-members:
   :no-inherited-members:

.. currentmodule:: pymare.core

.. autosummary:: pymare.core
   :toctree: generated/
   :template: class.rst

   pymare.core.Dataset

   :template: function.rst

   pymare.core.meta_regression


.. _api_estimators_ref:

:mod:`pymare.estimators`: Meta-analytic algorithms
--------------------------------------------------

.. automodule:: pymare.estimators
   :no-members:
   :no-inherited-members:

.. currentmodule:: pymare.estimators

.. autosummary:: pymare.estimators
   :toctree: generated/
   :template: class.rst

   pymare.estimators.WeightedLeastSquares
   pymare.estimators.DerSimonianLaird
   pymare.estimators.VarianceBasedLikelihoodEstimator
   pymare.estimators.SampleSizeBasedLikelihoodEstimator
   pymare.estimators.StanMetaRegression
   pymare.estimators.Hedges
   pymare.estimators.Stouffers
   pymare.estimators.Fishers


.. _api_results_ref:

:mod:`pymare.results`: Meta-analytic results
------------------------------------------------------

.. automodule:: pymare.results
   :no-members:
   :no-inherited-members:

.. currentmodule:: pymare.results

.. autosummary:: pymare.results
   :toctree: generated/
   :template: class.rst

   pymare.results.MetaRegressionResults
   pymare.results.CombinationTestResults
   pymare.results.PermutationTestResults
   pymare.results.BayesianMetaRegressionResults


.. _api_effectsize_ref:

:mod:`pymare.effectsize`: Effect size computation/conversion
------------------------------------------------------------

.. automodule:: pymare.effectsize
   :no-members:
   :no-inherited-members:

.. currentmodule:: pymare.effectsize

.. autosummary:: pymare.effectsize
   :toctree: generated/
   :template: class.rst

   pymare.effectsize.OneSampleEffectSizeConverter
   pymare.effectsize.TwoSampleEffectSizeConverter
   pymare.effectsize.Expression

   :template: function.rst
   pymare.effectsize.solve_system
   pymare.effectsize.Expression
   pymare.effectsize.select_expressions
   pymare.effectsize.compute_measure

.. _api_stats_ref:

:mod:`pymare.stats`: Miscellaneous statistical functions
--------------------------------------------------------

.. automodule:: pymare.stats
   :no-members:
   :no-inherited-members:

.. currentmodule:: pymare.stats

.. autosummary:: pymare.stats
   :toctree: generated/
   :template: function.rst

   pymare.stats.weighted_least_squares
   pymare.stats.ensure_2d
   pymare.stats.q_profile
   pymare.stats.q_gen
