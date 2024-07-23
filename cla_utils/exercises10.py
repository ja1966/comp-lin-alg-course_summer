import numpy as np
import numpy.random as random
import cla_utils


def arnoldi(A, b, k):
    """
    For a matrix A, apply k iterations of the Arnoldi algorithm,
    using b as the first basis vector.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array, the starting vector
    :param k: integer, the number of iterations

    :return Q: an mx(k+1) dimensional numpy array containing the orthonormal basis
    :return H: a (k+1)xk dimensional numpy array containing the upper \
    Hessenberg matrix
    """
    m = np.size(A, 0)
    Q = np.zeros((m, k+1), dtype=complex)
    H = np.zeros((k+1, k), dtype=complex)
    Q[:, 0] = b / np.linalg.norm(b)

    for i in range(k):
        v = A @ Q[:, i]
        H[:i+1, i] = Q[:, :i+1].conj().T @ v
        v -= Q[:, :i+1] @ H[:i+1, i]
        H[i+1, i] = np.linalg.norm(v)
        Q[:, i+1] = v / np.linalg.norm(v)

    return Q, H


def GMRES(A, b, maxit, tol, return_residual_norms=False,
          return_residuals=False):
    """
    For a matrix A, solve Ax=b using the basic GMRES algorithm.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array
    :param maxit: integer, the maximum number of iterations
    :param tol: floating point number, the tolerance for termination
    :param return_residual_norms: logical
    :param return_residuals: logical

    :return x: an m dimensional numpy array, the solution
    :return nits: if converged, the number of iterations required, otherwise \
    equal to -1
    :return rnorms: nits dimensional numpy array containing the norms of \
    the residuals at each iteration
    :return r: mxnits dimensional numpy array, column k contains residual \
    at iteration k
    """

    residual = 1
    nits = 0
    m = np.size(A, 0)
    Q = np.zeros((m, maxit+1), dtype=complex)
    H = np.zeros((maxit+1, maxit), dtype=complex)
    Q[:, 0] = b / np.linalg.norm(b)

    rnorms = np.zeros(maxit, dtype=complex)
    r = np.zeros((maxit, maxit), dtype=complex)

    while residual >= tol and nits <= maxit:

        # Step n of Arnoldi
        v = A @ Q[:, nits]
        H[:nits+1, nits] = Q[:, :nits+1].conj().T @ v
        v -= Q[:, :nits+1] @ H[:nits+1, nits]
        H[nits+1, nits] = np.linalg.norm(v)
        Q[:, nits+1] = v / np.linalg.norm(v)

        # Finding y to minimise the residual
        e_1 = np.concatenate([np.ones(1), np.zeros(nits+1)], dtype=complex)
        y = cla_utils.householder_ls(H[:nits+2, :nits+1], (np.linalg.norm(b) * e_1))

        x_n = Q[:nits+1, :nits+1] @ y

        r[:nits+2, nits] = (H[:nits+2, :nits+1] @ y) - np.linalg.norm(b) * e_1
        residual = np.linalg.norm(r[:, nits])
        rnorms[nits] = residual
        nits += 1

    # To check that rnorms and r return the correct matrices for the coursework

    if nits <= maxit:
        rnorms = rnorms[:nits]
        r = r[:nits+2, :nits+1]
        nits = -1

    if return_residual_norms and return_residuals:
        return x_n, nits, rnorms, r
    elif return_residual_norms:
        return x_n, nits, rnorms
    elif return_residuals:
        return x_n, nits, r
    else:
        return x_n, nits


def get_AA100():
    """
    Get the AA100 matrix.

    :return A: a 100x100 numpy array used in exercises 10.
    """
    AA100 = np.fromfile('AA100.dat', sep=' ')
    AA100 = AA100.reshape((100, 100))
    return AA100


def get_BB100():
    """
    Get the BB100 matrix.

    :return B: a 100x100 numpy array used in exercises 10.
    """
    BB100 = np.fromfile('BB100.dat', sep=' ')
    BB100 = BB100.reshape((100, 100))
    return BB100


def get_CC100():
    """
    Get the CC100 matrix.

    :return C: a 100x100 numpy array used in exercises 10.
    """
    CC100 = np.fromfile('CC100.dat', sep=' ')
    CC100 = CC100.reshape((100, 100))
    return CC100
