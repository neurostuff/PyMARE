"""Tests for pymare.stats."""
from pymare import stats


def test_q_gen(vars_with_intercept):
    """Test pymare.stats.q_gen."""
    result = stats.q_gen(*vars_with_intercept, 8)
    assert round(result[0], 4) == 8.0161


def test_q_profile(vars_with_intercept):
    """Test pymare.stats.q_profile."""
    bounds = stats.q_profile(*vars_with_intercept, 0.05)
    assert set(bounds.keys()) == {"ci_l", "ci_u"}
    assert round(bounds["ci_l"], 4) == 3.8076
    assert round(bounds["ci_u"], 2) == 59.61


def test_var_to_ci():
    """Test pymare.stats.var_to_ci.

    This is basically a smoke test. We should improve it.
    """
    ci = stats.var_to_ci(0.05, 0.5, n=20)
    assert round(ci[0], 4) == -0.2599
    assert round(ci[1], 4) == 0.3599
