API
===

.. _api_core_ref:

:mod:`pymare.core`: Core objects
--------------------------------------------------

.. automodule:: pymare.core
   :no-members:
   :no-inherited-members:

.. currentmodule:: pymare

.. autosummary::
   :toctree: generated/
   :template: class.rst

   core.Dataset

   :template: function.rst

   core.meta_regression


.. _api_estimators_ref:

:mod:`pymare.estimators`: Meta-analytic algorithms
--------------------------------------------------

.. automodule:: pymare.estimators
   :no-members:
   :no-inherited-members:

.. currentmodule:: pymare

.. autosummary::
   :toctree: generated/
   :template: class.rst

   estimators.WeightedLeastSquares
   estimators.DerSimonianLaird
   estimators.VarianceBasedLikelihoodEstimator
   estimators.SampleSizeBasedLikelihoodEstimator
   estimators.StanMetaRegression
   estimators.Hedges
   estimators.Stouffers
   estimators.Fishers


.. _api_results_ref:

:mod:`pymare.results`: Meta-analytic results
------------------------------------------------------

.. automodule:: pymare.results
   :no-members:
   :no-inherited-members:

.. currentmodule:: pymare

.. autosummary::
   :toctree: generated/
   :template: class.rst

   results.MetaRegressionResults
   results.CombinationTestResults
   results.PermutationTestResults
   results.BayesianMetaRegressionResults


.. _api_effectsize_ref:

:mod:`pymare.effectsize`: Effect size computation/conversion
------------------------------------------------------------

.. automodule:: pymare.effectsize
   :no-members:
   :no-inherited-members:

.. currentmodule:: pymare

.. autosummary::
   :toctree: generated/
   :template: class.rst

   effectsize.OneSampleEffectSizeConverter
   effectsize.TwoSampleEffectSizeConverter
   effectsize.Expression

   :template: function.rst
   effectsize.solve_system
   effectsize.Expression
   effectsize.select_expressions
   effectsize.compute_measure

.. _api_stats_ref:

:mod:`pymare.stats`: Miscellaneous statistical functions
--------------------------------------------------------

.. automodule:: pymare.stats
   :no-members:
   :no-inherited-members:

.. currentmodule:: pymare

.. autosummary::
   :toctree: generated/
   :template: function.rst

   stats.weighted_least_squares
   stats.ensure_2d
   stats.q_profile
   stats.q_gen
