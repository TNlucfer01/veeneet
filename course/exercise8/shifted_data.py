"""
shifted_data.py -- make_data.py extended with test points from other x-distributions.

../make_data.py is left untouched: the true function and the noise level are
taken from it, only the distribution that x is drawn from changes.
"""
import inspect
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import make_data

NOISE = inspect.signature(make_data.make).parameters["noise"].default


def _y(x, rng):
    return make_data.true_function(x) + NOISE * rng.standard_normal(len(x))


def uniform(n, rng):
    """Same distribution as the training data: x ~ U[-2, 2]."""
    x = rng.uniform(-2.0, 2.0, size=n)
    return x, _y(x, rng)


def edges(n, rng):
    """Inside [-2, 2] but piled up near the ends: x = 2(2B - 1), B ~ Beta(0.5, 0.5)."""
    x = 2.0 * (2.0 * rng.beta(0.5, 0.5, size=n) - 1.0)
    return x, _y(x, rng)


def outside(n, rng, width=0.5):
    """Just outside the training range: |x| ~ U[2, 2 + width], random sign."""
    x = rng.choice([-1.0, 1.0], size=n) * rng.uniform(2.0, 2.0 + width, size=n)
    return x, _y(x, rng)
