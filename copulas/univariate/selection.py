"""Univariate selection function."""

import numpy as np
from scipy.stats import kstest

from copulas import get_instance


def select_univariate(X, candidates,return_names=False):
    """Select the best univariate class for this data.

    Args:
        X (pandas.DataFrame):
            Data for which be best univariate must be found.
        candidates (list[Univariate]):
            List of Univariate subclasses (or instances of those) to choose from.

    Returns:
        Univariate:
            Instance of the selected candidate.
    """
    best_ks = np.inf
    best_model = None
    for model in candidates:
        try:
            instance = get_instance(model)
            instance.fit(X)
            ks, _ = kstest(X, instance.cdf)
            if ks < best_ks:
                best_ks = ks
                best_model = model
                model_name = model.__module__
        except Exception:
            # Distribution not supported
            pass
    
    if return_names:
        return get_instance(best_model), model_name
    
    else :
        return get_instance(best_model)
