import numpy as np


def householder(A, kmax=None):
    """
    Given a real mxn matrix A, find the reduction to upper triangular matrix R
    using Householder transformations. The reduction should be done "in-place",
    so that A is transformed to R.

    :param A: an mxn-dimensional numpy array
    :param kmax: an integer, the number of columns of A to reduce \
    to upper triangular. If not present, will default to n.
    """

    m, n = A.shape
    if kmax is None:
        kmax = n

    for k in range(kmax):
        x = np.copy(A[k:, k])
        if np.sign(x[0]) == 0:
            x[0] += np.linalg.norm(x)
        else:
            x[0] += np.sign(x[0]) * np.linalg.norm(x)

        x /= np.linalg.norm(x)
        A[k:, k:] -= 2 * np.outer(x, np.inner(x.conj(), A[k:, k:].T))


def solve_U(U, b):
    """
    Solve systems Ux_i=b_i for x_i with U upper triangular, i=1,2,...,k

    :param U: an mxm-dimensional numpy array, assumed upper triangular
    :param b: an mxk-dimensional numpy array, with ith column containing
       b_i
    :return x: an mxk-dimensional numpy array, with ith column containing
       the solution x_i

    """
    m, k = np.shape(b)
    x = np.zeros((m, k), dtype=float)
    x[m-1, :] = b[m-1, :] / U[m-1, m-1]

    for i in range(m-2, -1, -1):
        x[i, :] = (b[i, :] - U[i, i+1:] @ x[i+1:, :]) / U[i, i]

    return x


def householder_solve(A, b):
    """
    Given a real mxm matrix A, use the Householder transformation to solve
    Ax_i=b_i, i=1,2,...,k.

    :param A: an mxm-dimensional numpy array
    :param b: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors b_1,b_2,...,b_k.

    :return x: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors x_1,x_2,...,x_k.
    """

    m = np.size(b, 0)
    A_hat = np.hstack([A, b])
    householder(A_hat, kmax=m)
    x = solve_U(A_hat[:, :m], A_hat[:, m:])

    return x


def householder_qr(A):
    """
    Given a real mxn matrix A, use the Householder transformation to find
    the full QR factorisation of A.

    :param A: an mxn-dimensional numpy array

    :return Q: an mxm-dimensional numpy array
    :return R: an mxn-dimensional numpy array
    """
    m, n = np.shape(A)
    A = np.hstack([A, np.identity(m)])
    householder(A, kmax=n)
    R = A[:, :n]
    Q = A[:, n:].T

    return Q, R


def householder_ls(A, b):
    """
    Given a real mxn matrix A and an m dimensional vector b, find the
    least squares solution to Ax = b.

    :param A: an mxn-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an n-dimensional numpy array
    """

    m, n = np.shape(A)
    A_hat = np.hstack([A, b.reshape(m, 1)])
    householder(A_hat, kmax=n)
    R_hat = A_hat[:n, :n]
    b_hat = A_hat[:n, -1]
    x = solve_U(R_hat, b_hat.reshape((np.size(b_hat, 0), 1))).reshape(n)

    return x
