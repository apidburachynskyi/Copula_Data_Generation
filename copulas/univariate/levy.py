"""LevyUnivariate module."""

# from scipy.stats import norm
from scipy.stats import levy

from copulas.univariate.base import BoundedType, ParametricType, ScipyModel


class LevyUnivariate(ScipyModel):
    """Levy univariate model."""

    PARAMETRIC = ParametricType.PARAMETRIC
    BOUNDED = BoundedType.UNBOUNDED

    MODEL_CLASS = levy

    def _fit(self,X):
        params = levy.fit(X)
        self._params = {
            'loc': params[0],
            'scale': params[1]
        }