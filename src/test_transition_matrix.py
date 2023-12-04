"""tests for transition matrices ensemble generator"""

import numpy as np
from hamiltonian_utils import get_random_isings_ensemble  # type: ignore # pylint: disable = no-name-in-module
from transition_matrix import _get_squared_unitary, get_transition_matrix
from test_utils import std_normal_ensemble, uniform_ensemble


def test_get_squared_uniform():
    """tests a double stochastic propertie of generated U ** 2 matrix"""
    ensemble = get_random_isings_ensemble(
        100,
        5,
        1.0,
        *uniform_ensemble,
    )
    transition = _get_squared_unitary(ensemble, 0.6, 0.3)
    assert np.isclose(transition.sum(1), np.ones((100, 32))).all()
    assert np.isclose(transition.sum(2), np.ones((100, 32))).all()


def test_get_transition_matrix():
    """tests transition matrix's stochastic propertie and
    stationary state property"""
    ensemble = get_random_isings_ensemble(
        100,
        5,
        1.0,
        *std_normal_ensemble,
    )
    transition = get_transition_matrix(ensemble, 0.6, 0.3)
    assert (transition > 0).all()
    assert np.isclose(transition.sum(1), np.ones((100, 32))).all()
    lmbd, v = np.linalg.eig(transition)
    indices = np.argmax(np.abs(lmbd), axis=1)
    stat_points = v[np.arange(0, 100), :, indices]
    stat_points /= stat_points.sum(1, keepdims=True)
    distr = np.exp(-ensemble.get_diagonals())
    distr /= distr.sum(1, keepdims=True)
    assert np.isclose(distr, stat_points).all()
