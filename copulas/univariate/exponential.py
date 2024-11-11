"""ExponentialUnivariate module."""

# from scipy.stats import norm
from scipy.stats import expon
import numpy as np

from copulas.univariate.base import BoundedType, ParametricType, ScipyModel


class ExponentialUnivariate(ScipyModel):
    """Exponential univariate model."""

    PARAMETRIC = ParametricType.PARAMETRIC
    BOUNDED = BoundedType.UNBOUNDED

    MODEL_CLASS = expon

    def _fit(self, X):
        params = expon.fit(X)
        self._params = {
            'loc': params[0],
            'scale': params[1]
        }

    # def _is_constant(self):
        # return self._params['scale'] == 0

    # def _extract_constant(self):
        # return self._params['???']
