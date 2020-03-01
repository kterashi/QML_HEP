import numpy as np
from functools import reduce
from qulacs.gate import X, Z, DenseMatrix


# Basic gate
I_mat = np.eye(2, dtype=complex)
X_mat = X(0).get_matrix()
Z_mat = Z(0).get_matrix()


## Function to create full-size gate
def make_fullgate(list_SiteAndOperator, nqubit):
    """
    Receive list_SiteAndOperator = [ [i_0, O_0], [i_1, O_1], ...]
    Insert Identity to irrelevant qubtis
    Create (2**nqubit, 2**nqubit) martrix of I(0) * ... * O_0(i_0) * ... * O_1(i_1) ...
    """
    list_Site = [SiteAndOperator[0] for SiteAndOperator in list_SiteAndOperator]
    list_SingleGates = []  ## reduce 1-qubit gates using np.kron
    cnt = 0
    for i in range(nqubit):
        if i in list_Site:
            list_SingleGates.append( list_SiteAndOperator[cnt][1] )
            cnt += 1
        else:
            list_SingleGates.append(I_mat)

    return reduce(np.kron, list_SingleGates)


def create_time_evol_gate(nqubit, time_step=0.77):
    """ Ising hamiltonian with random magnetic field and random couplings
    :param time_step: elapsed time of random hamiltonian evolution
    :return  qulacs gate object
    """
    ham = np.zeros((2**nqubit,2**nqubit), dtype = complex)
    for i in range(nqubit):  # i runs 0 to nqubit-1
        Jx = -1. + 2.*np.random.rand()  # [-1,1]
        ham += Jx * make_fullgate( [ [i, X_mat] ], nqubit)
        for j in range(i+1, nqubit):
            J_ij = -1. + 2.*np.random.rand()
            ham += J_ij * make_fullgate ([ [i, Z_mat], [j, Z_mat]], nqubit)

    ## Build time-evolution operator by diagonalizing the Ising hamiltonian H*P = P*D <-> H = P*D*P^dagger
    diag, eigen_vecs = np.linalg.eigh(ham)
    time_evol_op = np.dot(np.dot(eigen_vecs, np.diag(np.exp(-1j*time_step*diag))), eigen_vecs.T.conj())  # e^-iHT

    # Convert to qulacs gate
    time_evol_gate = DenseMatrix([i for i in range(nqubit)], time_evol_op)

    return time_evol_gate


def min_max_scaling(x, axis=None):
    """Normalized to [-1, 1]"""
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    result = 2.*result-1.
    return result

# KT
def min_max_scaling_0to2pi(x, axis=None):
    """Normalized to [0, 2pi]"""
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    result = 2.*np.pi*result
    return result

def softmax(x):
    """softmax function
    :param x: ndarray
    """
    exp_x = np.exp(x)
    y = exp_x / np.sum(np.exp(x))
    return y
