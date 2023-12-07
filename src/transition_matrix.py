"""utils for generaing quantum accelerated MCMC transition matrices from agiven
ensemble"""

import multiprocessing
from os import environ
import numpy as np
from hamiltonian_utils import IsingsEnsemble, get_local_updates_transition_matrix  # type: ignore # pylint: disable = no-name-in-module
from xhamiltonian import _get_x_hamiltonian

environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())


def _get_diagonal_matrix(
    array: np.ndarray,
) -> np.ndarray:
    n = array.shape[-1]
    out = np.zeros((*array.shape[:-1], n, n), dtype=array.dtype)
    out.reshape(-1, n**2)[..., :: n + 1] = array
    return out


def _get_squared_unitary(
    ensemble: IsingsEnsemble,
    tau: float,
    gamma: float,
) -> np.ndarray:
    spins_number = ensemble.get_spins_number()
    diagonals = ensemble.get_diagonals()
    x_hamiltonian = _get_x_hamiltonian(spins_number)
    alphas = ensemble.get_normalizing_factors()[..., np.newaxis, np.newaxis]
    hamiltonian = (
        _get_diagonal_matrix(diagonals) * alphas * (1 - gamma) + x_hamiltonian * gamma
    )
    lmbd, u = np.linalg.eigh(hamiltonian)
    exp_lmbd = np.exp(-1j * tau * lmbd)
    unitary = u @ (exp_lmbd[..., np.newaxis] * u.swapaxes(-2, -1).conj())
    unitary_sq = unitary * unitary.conj()
    return unitary_sq


def get_transition_matrix(
    ensemble: IsingsEnsemble,
    tau: float,
    gamma: float,
    inv_temperature: float,
) -> np.ndarray:
    """Returns a set of Quantum accelerated MCMC transition matrices
    randomly generated from a given Ising ensemble.

    Args:
        ensemble: ising ensemble
        tau: time step
        gamma: a parameter determining weight of a quantum Hamiltonian parts, i.e.
            (ZZ * (1 - gamma) + X * gamma)
        inv_temperature: a inv_temperature of the ising task

    Returns:
        a np.ndarray of shape (ensemble_size, 2 ** spins_number, 2 ** spin_number)"""

    sq_unitary = _get_squared_unitary(ensemble, tau, gamma)
    diagonals = ensemble.get_diagonals()
    correction = np.exp(
        inv_temperature * (diagonals[..., np.newaxis, :] - diagonals[..., np.newaxis])
    )
    correction = np.minimum(1, correction)
    transition = sq_unitary * correction
    new_diag = 1 - transition.sum(-2) + np.diagonal(transition, axis1=-2, axis2=-1)
    transition.reshape((-1, transition.shape[-1] ** 2))[
        ..., :: (transition.shape[-1] + 1)
    ] = new_diag
    return transition


def get_local_transition_matrix(
    ensemble: IsingsEnsemble,
    inv_temperature: float,
) -> np.ndarray:
    """Retuens transition matrices corresponding local wrt to the hamming distance
    updates.
    Args:
        ensemble: ising ensemble
        inv_temperature: inverse temperature of an ising ensemble"""
    spins_number = ensemble.get_spins_number()
    size = 2**spins_number
    diagonals = ensemble.get_diagonals()
    raw_transition = np.zeros((size, size))
    get_local_updates_transition_matrix(raw_transition)
    assert (raw_transition >= 0).all()
    assert np.isclose(np.sum(raw_transition, axis=-2), 1.0).all()
    correction = np.exp(
        inv_temperature * (diagonals[..., np.newaxis, :] - diagonals[..., np.newaxis])
    )
    correction = np.minimum(1, correction)
    transition = raw_transition * correction
    new_diag = 1 - transition.sum(-2) + np.diagonal(transition, axis1=-2, axis2=-1)
    transition.reshape((-1, transition.shape[-1] ** 2))[
        ..., :: (transition.shape[-1] + 1)
    ] = new_diag
    return transition


def get_uniform_transition_matrix(
    ensemble: IsingsEnsemble,
    inv_temperature: float,
) -> np.ndarray:
    """Retuens transition matrices corresponding to uniformly distributed updates.
    Args:
        ensemble: ising ensemble
        inv_temperature: inverse temperature of an ising ensemble"""
    spins_number = ensemble.get_spins_number()
    size = 2**spins_number
    diagonals = ensemble.get_diagonals()
    raw_transition = np.ones((size, size))
    raw_transition -= np.diag(np.diag(raw_transition))
    raw_transition /= raw_transition.sum(1, keepdims=True)
    assert (raw_transition >= 0).all()
    assert np.isclose(np.sum(raw_transition, axis=-2), 1.0).all()
    correction = np.exp(
        inv_temperature * (diagonals[..., np.newaxis, :] - diagonals[..., np.newaxis])
    )
    correction = np.minimum(1, correction)
    transition = raw_transition * correction
    new_diag = 1 - transition.sum(-2) + np.diagonal(transition, axis1=-2, axis2=-1)
    transition.reshape((-1, transition.shape[-1] ** 2))[
        ..., :: (transition.shape[-1] + 1)
    ] = new_diag
    return transition
