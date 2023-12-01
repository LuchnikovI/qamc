import numpy as np
from numpy.typing import ArrayLike

sx = np.array([[0, 1], [1, 0]], dtype=np.float64)

def _get_x_hamiltonian(qubits_number: int) -> ArrayLike:
    x_hamiltonian = np.kron(sx, np.eye(2 ** (qubits_number - 1)))
    for i in range(1, qubits_number):
        x_hamiltonian += np.kron(np.kron(np.eye(2 ** i), sx), np.eye(2 ** (qubits_number - 1 - i)))
    return x_hamiltonian
