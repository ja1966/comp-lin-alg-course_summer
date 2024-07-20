import numpy as np
import cla_utils


def Q1AQ1s(A):
    """
    For a matrix A, find the unitary matrix Q1 such that the first
    column of Q1*A has zeros below the diagonal. Then return A1 = Q1*A*Q1^*.

    :param A: an mxm numpy array

    :return A1: an mxm numpy array
    """

    cla_utils.householder(A, kmax=0)
    A = A.conj().T
    cla_utils.householder(A, kmax=0)
    A = A.conj().T

    return A


def hessenberg(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place.

    :param A: an mxm numpy array
    """

    m = np.size(A, 0)

    for k in range(m-2):
        x = np.copy(A[k+1:, k])

        if np.sign(x[0]) == 0:
            x[0] += np.linalg.norm(x)
        else:
            x[0] += np.sign(x[0]) * np.linalg.norm(x)

        x /= np.linalg.norm(x)
        A[k+1:, k:] -= 2 * np.outer(x, np.inner(x.conj(), A[k+1:, k:].T))
        A[:, k+1:] -= 2 * np.outer(A[:, k+1:] @ x, x.conj().T)


def hessenbergQ(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place, and return the matrix Q
    for which QHQ^* = A.

    :param A: an mxm numpy array

    :return Q: an mxm numpy array
    """

    m = np.size(A, 0)
    Q = np.identity(m)
    A0 = np.copy(A)

    for k in range(m):
        x_Q = np.copy(A0[k:, k])

        if np.sign(x_Q[0]) == 0:
            x_Q[0] += np.linalg.norm(x_Q)
        else:
            x_Q[0] += np.sign(x_Q[0]) * np.linalg.norm(x_Q)

        x_Q /= np.linalg.norm(x_Q)
        Q[k:, k:] = Q[k:, k:] @ (np.identity(m-k) - 2 * np.outer(x_Q, x_Q.conj().T))

    for k in range(m-2):
        x = np.copy(A[k+1:, k])

        if np.sign(x[0]) == 0:
            x[0] += np.linalg.norm(x)
        else:
            x[0] += np.sign(x[0]) * np.linalg.norm(x)

        x /= np.linalg.norm(x)

        A[k+1:, k:] -= 2 * np.outer(x, np.inner(x.conj(), A[k+1:, k:].T))
        A[:, k+1:] -= 2 * np.outer(A[:, k+1:] @ x, x.conj().T)

    return Q


def hessenberg_ev(H):
    """
    Given a Hessenberg matrix, return the eigenvectors.

    :param H: an mxm numpy array

    :return V: an mxm numpy array whose columns are the eigenvectors of H

    Do not change this function.
    """
    m, n = H.shape
    assert(m==n)
    assert(cla_utils.norm(H[np.tril_indices(m, -2)]) < 1.0e-6)
    _, V = np.linalg.eig(H)
    return V


def ev(A):
    """
    Given a matrix A, return the eigenvectors of A. This should
    be done by using your functions to reduce to upper Hessenberg
    form, before calling hessenberg_ev (which you should not edit!).

    :param A: an mxm numpy array

    :return V: an mxm numpy array whose columns are the eigenvectors of A
    """

    raise NotImplementedError
