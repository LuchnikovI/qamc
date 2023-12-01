import numpy as np
from transition_matrix import _get_squared_unitary, get_transition_matrix
from hamiltonian_utils import get_random_isings_ensemble

def test_get_squared_uniform():
    ensemble = get_random_isings_ensemble(
        100, 5, 1., "uniform", {"lb": 0.5, "ub": 1.5}, "uniform", {"lb": 0.5, "ub": 1.5},
    )
    transition = _get_squared_unitary(ensemble, 0.6, 0.3)
    assert np.isclose(transition.sum(1), np.ones((100, 32))).all()
    assert np.isclose(transition.sum(2), np.ones((100, 32))).all()

def test_get_transition_matrix():
    ensemble = get_random_isings_ensemble(
        100, 5, 1., "normal", {"mu": 0., "std": 1.}, "normal", {"mu": 0., "std": 1.},
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
