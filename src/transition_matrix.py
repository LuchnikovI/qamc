import multiprocessing
from os import environ
environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
import numpy as np
from numpy.typing import ArrayLike
from xhamiltonian import _get_x_hamiltonian
from hamiltonian_utils import get_random_isings_ensemble, IsingsEnsemble

def _get_diagonal_matrix(
        array: ArrayLike,
) -> ArrayLike:
    n = array.shape[-1]
    out = np.zeros((*array.shape[:-1], n, n), dtype=array.dtype)
    out.reshape(-1, n ** 2)[...,::n + 1] = array
    return out

def _get_squared_unitary(
        ensemble: IsingsEnsemble,
        tau: float,
        gamma: float,
) -> ArrayLike:
    spins_number = ensemble.get_spins_number()
    diagonals = ensemble.get_diagonals()
    x_hamiltonian = _get_x_hamiltonian(spins_number)
    alphas = ensemble.get_normalizing_factors()[..., np.newaxis, np.newaxis]
    hamiltonian = _get_diagonal_matrix(diagonals) * alphas * (1 - gamma) + x_hamiltonian * gamma
    lmbd, u = np.linalg.eigh(hamiltonian)
    exp_lmbd = np.exp(-1j * tau * lmbd)
    unitary = u @ (exp_lmbd[..., np.newaxis] * u.swapaxes(-2, -1).conj())
    unitary_sq = unitary * unitary.conj()
    return unitary_sq

def get_transition_matrix(
        ensemble: IsingsEnsemble,
        tau: float,
        gamma: float,
) -> ArrayLike:
    sq_unitary = _get_squared_unitary(ensemble, tau, gamma)
    diagonals = ensemble.get_diagonals()
    correction = np.exp(diagonals[..., np.newaxis, :] - diagonals[..., np.newaxis])
    correction = np.minimum(1, correction)
    transition = sq_unitary * correction
    new_diag = 1 - transition.sum(-2) + np.diagonal(transition, axis1=-2, axis2=-1)
    transition.reshape((-1, transition.shape[-1] ** 2))[..., ::(transition.shape[-1] + 1)] = new_diag
    return transition
