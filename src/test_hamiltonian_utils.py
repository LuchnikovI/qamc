"""tests for utils aimed on hamiltonians construction"""

import numpy as np
import pytest
from hamiltonian_utils import get_random_isings_ensemble, IsingsEnsemble  # type: ignore # pylint: disable = no-name-in-module
from test_utils import (
    uniform_ensemble,
    discrete_ensemble,
    std_normal_ensemble,
    Ensemble,
)


@pytest.mark.parametrize(
    "ensemble_type",
    [
        std_normal_ensemble,
        uniform_ensemble,
        discrete_ensemble,
    ],
)
@pytest.mark.parametrize("density", [1.0, 0.5, 0.0])
@pytest.mark.parametrize("spins_number", [1, 5, 10])
@pytest.mark.parametrize("ensemble_size", [1, 128])
def test_random_isings_ensemble_shape(
    ensemble_type: Ensemble,
    density: float,
    spins_number: int,
    ensemble_size: int,
):
    """tests shape of the diagonal of a randomly generated ising ensemble"""
    diag_size = 2**spins_number
    ensemble = get_random_isings_ensemble(
        ensemble_size,
        spins_number,
        density,
        *ensemble_type,
    )
    diagonals = ensemble.get_diagonals()
    assert (diagonals < 1e5).all()
    assert (diagonals > -1e5).all()
    assert diagonals.shape == (ensemble_size, diag_size)


def test_isings_ensemble_small():
    """tests some elements of the diagonal of a randomly generated ising ensemble"""
    ensemble = IsingsEnsemble(
        [
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
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
    )
    diagonals = ensemble.get_diagonals()
    assert np.isclose(diagonals[0, 0], -6.3)
    assert np.isclose(diagonals[0, -1], -2.3)
    assert np.isclose(diagonals[1, 2], -3.3)
    assert np.isclose(diagonals[1, -3], -3.3)


@pytest.mark.parametrize(
    "ensemble_type",
    [
        std_normal_ensemble,
        uniform_ensemble,
        discrete_ensemble,
    ],
)
@pytest.mark.parametrize("density", [1.0, 0.5, 0.0])
@pytest.mark.parametrize("spins_number", [1, 50, 100])
@pytest.mark.parametrize("ensemble_size", [1, 128])
def test_normalization_constants_shape(
    ensemble_type: Ensemble,
    density: float,
    spins_number: int,
    ensemble_size: int,
):
    """tests correctnes of normalization constants shape"""
    ensemble = get_random_isings_ensemble(
        ensemble_size,
        spins_number,
        density,
        *ensemble_type,
    )
    alphas = ensemble.get_normalizing_factors()
    assert (alphas > -1e5).all()
    assert (alphas < 1e5).all()
    assert alphas.shape == (ensemble_size,)


def test_normalization_constants_small():
    """tests correction of normalization constants alpha from the paper"""
    ensemble = IsingsEnsemble(
        [
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
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
    )
    trial_alphas = ensemble.get_normalizing_factors()
    correct_alphas = np.array(
        [
            np.sqrt(3) / np.sqrt(2 * (1.1**2) + 2.1**2 + 2),
            np.sqrt(3) / np.sqrt(3 * (1.1**2)),
        ]
    )
    assert np.isclose(correct_alphas, trial_alphas).all()
