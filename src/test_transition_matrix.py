"""tests for transition matrices ensemble generator"""

import numpy as np
import pytest
from hamiltonian_utils import get_random_isings_ensemble, get_local_updates_transition_matrix  # type: ignore # pylint: disable = no-name-in-module
from transition_matrix import (
    _get_squared_unitary,
    get_transition_matrix,
    get_local_transition_matrix,
    get_uniform_transition_matrix,
)
from test_utils import (
    std_normal_ensemble,
    uniform_ensemble,
    discrete_ensemble,
    Ensemble,
)


@pytest.mark.parametrize(
    "spins_number",
    [2, 5, 7],
)
def test_get_local_updates_transition_matrix(
    spins_number: int,
):
    """tests stochastic properties of the local updates transition matrix"""
    size = 2**spins_number
    transition_matrix = np.zeros((size, size))
    get_local_updates_transition_matrix(transition_matrix)
    assert (transition_matrix >= 0).all()
    assert np.isclose(np.sum(transition_matrix, axis=0), 1.0).all()


@pytest.mark.parametrize(
    "ensemble_type",
    [
        std_normal_ensemble,
        uniform_ensemble,
        discrete_ensemble,
    ],
)
@pytest.mark.parametrize("density", [1.0, 0.5, 0.0])
def test_square_uniform(ensemble_type: Ensemble, density: float):
    """tests a double stochastic propertie of generated U ** 2 matrix"""
    ensemble = get_random_isings_ensemble(
        100,
        5,
        density,
        *ensemble_type,
    )
    transition = _get_squared_unitary(ensemble, 0.6, 0.3)
    assert np.isclose(transition.sum(1), np.ones((100, 32))).all()
    assert np.isclose(transition.sum(2), np.ones((100, 32))).all()


@pytest.mark.parametrize(
    "ensemble_type",
    [
        std_normal_ensemble,
        uniform_ensemble,
        discrete_ensemble,
    ],
)
@pytest.mark.parametrize("density", [1.0, 0.5, 0.0])
@pytest.mark.parametrize("inv_temperature", [0.2, 1.0, 5.0])
def test_get_transition_matrix(
    ensemble_type: Ensemble,
    density: float,
    inv_temperature: float,
):
    """tests transition matrix's stochastic propertie and
    stationary state property"""
    ensemble = get_random_isings_ensemble(
        100,
        5,
        density,
        *ensemble_type,
    )
    transition = get_transition_matrix(ensemble, 0.6, 0.3, inv_temperature)
    assert (transition > 0).all()
    assert np.isclose(transition.sum(1), np.ones((100, 32))).all()
    lmbd, v = np.linalg.eig(transition)
    indices = np.argmax(np.abs(lmbd), axis=1)
    stat_points = v[np.arange(0, 100), :, indices]
    stat_points /= stat_points.sum(1, keepdims=True)
    distr = np.exp(-ensemble.get_diagonals() * inv_temperature)
    distr /= distr.sum(1, keepdims=True)
    assert np.isclose(distr, stat_points, rtol=1e-6, atol=1e-6).all()


@pytest.mark.parametrize(
    "ensemble_type",
    [
        std_normal_ensemble,
        uniform_ensemble,
        discrete_ensemble,
    ],
)
@pytest.mark.parametrize("density", [1.0, 0.5, 0.0])
@pytest.mark.parametrize("inv_temperature", [0.2, 1.0, 2.0])
def test_get_local_transition_matrix(
    ensemble_type: Ensemble,
    density: float,
    inv_temperature: float,
):
    """tests transition matrix's stochastic propertie and
    stationary state property"""
    ensemble = get_random_isings_ensemble(
        100,
        5,
        density,
        *ensemble_type,
    )
    transition = get_local_transition_matrix(ensemble, inv_temperature)
    assert (transition >= 0.0).all()
    assert np.isclose(transition.sum(-2), np.ones((100, 32))).all()
    lmbd, v = np.linalg.eig(transition)
    indices = np.argmax(np.abs(lmbd), axis=1)
    stat_points = v[np.arange(0, 100), :, indices]
    stat_points /= stat_points.sum(1, keepdims=True)
    assert np.min(stat_points) > -1e-6
    distr = np.exp(-ensemble.get_diagonals() * inv_temperature)
    distr /= distr.sum(1, keepdims=True)
    assert np.max(np.abs(distr - stat_points)) < 1e-6


@pytest.mark.parametrize(
    "ensemble_type",
    [
        std_normal_ensemble,
        uniform_ensemble,
        discrete_ensemble,
    ],
)
@pytest.mark.parametrize("density", [1.0, 0.5, 0.0])
@pytest.mark.parametrize("inv_temperature", [0.2, 1.0, 2.0])
def test_get_uniform_transition_matrix(
    ensemble_type: Ensemble,
    density: float,
    inv_temperature: float,
):
    """tests transition matrix's stochastic propertie and
    stationary state property"""
    ensemble = get_random_isings_ensemble(
        100,
        5,
        density,
        *ensemble_type,
    )
    transition = get_uniform_transition_matrix(ensemble, inv_temperature)
    assert (transition >= 0.0).all()
    assert np.isclose(transition.sum(-2), np.ones((100, 32))).all()
    lmbd, v = np.linalg.eig(transition)
    indices = np.argmax(np.abs(lmbd), axis=1)
    stat_points = v[np.arange(0, 100), :, indices]
    stat_points /= stat_points.sum(1, keepdims=True)
    assert np.min(stat_points) > -1e-6
    distr = np.exp(-ensemble.get_diagonals() * inv_temperature)
    distr /= distr.sum(1, keepdims=True)
    assert np.max(np.abs(distr - stat_points)) < 1e-6
