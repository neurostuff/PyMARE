"""Statistical expressions."""

from sympy import sympify


class Expression:
    """Represents a single statistical expression.
    
    Args:
        expr (str): String representation of the mathematical expression.
        description (str, optional): Optional text description of expression.
        inputs (bool, optional): Indicates whether the expression applies
            in the one-sample case (1), two-sample case (2), or both
            (None).
        metric (str, optional): Name of metric for which this expression
            applies. Defaults to None (all inputs with unlabeled metricss).
    """
    def __init__(self, expr, description=None, inputs=None, metric=None):
        self.expr = expr
        self.description = description
        self.inputs = inputs
        self.metric = metric

        self.sympy = sympify(expr)
        self.symbols = self.sympy.free_symbols


expressions = [
    Expression('sd - sqrt(v)'),
    Expression('sem - sd / sqrt(n)'),
    Expression('sd2 - sqrt(v2)', inputs=2),
    Expression('sd_pooled - sqrt((v + v2) / 2)', inputs=2),
    Expression('t - y / sem', "One-sample t-test", inputs=1),
    Expression('d - y / sd', "Cohen's d (one sample)", inputs=1),
    Expression('d - t / sqrt(n)', "Cohen's d (from t)", inputs=1),
    Expression('g - d * j', "Hedges' g"),
    # TODO: we currently use Hedges' approximation instead of original J
    # function because the gamma function slows solving down considerably and
    # breaks numpy during lambdification. Need to fix/improve this.
    Expression('j - (1 - (3 / (4 * (n - 1) - 1)))', "Approximate correction "
               "factor for Hedges' g", inputs=1),
    Expression('t - (y - y2) / sqrt(v / n + v2 / n2)', "Two-sample t-test "
               "(unequal variances)", inputs=2),
    Expression('sd_pooled - sqrt((v * (n - 1) + v2 * (n2 - 1)) / (n + n2 + 2))',
               "Pooled standard deviation (Cohen version)", inputs=2),
    Expression('d - (y - y2) / sd_pooled', "Cohen's d (two-sample)",
               inputs=2),
    Expression('d - t * sqrt(1 / n + 1 / n2)', "Cohen's d (two-sample from t)",
               inputs=2),
    Expression('j - (1 - (3 / (4 * (n + n2) - 9)))', "Approximate correction "
               "factor for Hedges' g", inputs=2)
]
