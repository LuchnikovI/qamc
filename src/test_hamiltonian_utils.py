import numpy as np
from hamiltonian_utils import get_random_isings_ensemble, IsingsEnsemble

def test_random_isings_ensemble_shape():
    ensemble = get_random_isings_ensemble(
        2, 10, 1., "uniform", {"lb": 0.5, "ub": 1.5}, "uniform", {"lb": 0.5, "ub": 1.5},
    )
    diagonals = ensemble.get_diagonals()
    assert diagonals.shape == (2, 1024)
    ensemble = get_random_isings_ensemble(
        1000, 9, 1., "discrete", {}, "discrete", {},
    )
    diagonals = ensemble.get_diagonals()
    assert diagonals.shape == (1000, 512)

def test_isings_ensemble_small():
    ensemble = IsingsEnsemble([
        [
            (0, 1, 1.1),
            (1, 2, 2.1),
            (2, 0, 1.1),
        ],
        [
            (0, 1, -1.1),
            (1, 2, -1.1),
            (2, 0, 1.1),
        ],
    ],
    [
        [1., 0., 1.],
        [0., 0., 0.],
    ])
    diagonals = ensemble.get_diagonals()
    assert np.isclose(diagonals[0,  0], -6.3)
    assert np.isclose(diagonals[0,  -1], -2.3)
    assert np.isclose(diagonals[1,  2], -3.3)
    assert np.isclose(diagonals[1,  -3], -3.3)

def test_normalization_constants_shape():
    ensemble = get_random_isings_ensemble(
        10, 100, 1., "uniform", {"lb": 0.5, "ub": 1.5}, "uniform", {"lb": 0.5, "ub": 1.5},
    )
    alphas = ensemble.get_normalizing_factors()
    alphas.shape = (10,)

def test_normalization_constants_small():
    ensemble = IsingsEnsemble([
        [
            (0, 1, 1.1),
            (1, 2, 2.1),
            (2, 0, 1.1),
        ],
        [
            (0, 1, -1.1),
            (1, 2, -1.1),
            (2, 0, 1.1),
        ],
    ],
    [
        [1., 0., 1.],
        [0., 0., 0.],
    ])
    trial_alphas = ensemble.get_normalizing_factors()
    correct_alphas = np.array([
        np.sqrt(3) / np.sqrt(2 * (1.1 ** 2) + 2.1 ** 2 + 2),
        np.sqrt(3) / np.sqrt(3 * (1.1 ** 2))
    ])
    assert np.isclose(correct_alphas, trial_alphas).all()
