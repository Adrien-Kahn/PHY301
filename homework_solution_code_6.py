import numpy as np


def tensor_product(matrix_list):
    """Construct the Kronecker product of a list of matrices
    """
    current_matrix = 1
    for matrix in matrix_list:
        current_matrix = np.kron(current_matrix, matrix)
    return current_matrix


n_spins = 10

# Define the Pauli matrices
sigmax = np.array([[0,1],[1,0]])
sigmaz = np.array([[1,0],[0,-1]])


hamiltonian = 0

# Add the sigma_x terms
for k in range(n_spins):
    hamiltonian += -tensor_product([np.eye(2)]*k + [sigmax] + [np.eye(2)]*(n_spins-k-1))

# Add the sigma_z sigma_z terms
for k in range(n_spins-1):
    hamiltonian += tensor_product([np.eye(2)]*k + [sigmaz, sigmaz] + [np.eye(2)]*(n_spins-k-2))

# Get the lowest eigenvalue (np.linalg.eigh outputs them in increasing order)
print(np.linalg.eigh(hamiltonian)[0][0])