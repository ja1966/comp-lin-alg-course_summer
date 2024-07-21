import numpy as np
from time import time


def orthog_cpts(v, Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute v = r + u_1q_1 + u_2q_2 + ... + u_nq_n
    for scalar coefficients u_1, u_2, ..., u_n and
    residual vector r

    :param v: an m-dimensional numpy array
    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return r: an m-dimensional numpy array containing the residual
    :return u: an n-dimensional numpy array containing the coefficients
    """

    n = np.size(Q, 1)
    r = np.copy(v)
    u = np.zeros(n, dtype=complex)
    for i in range(n):
        u[i] = np.inner(Q[:, i].conj(), v)
        r -= u[i] * Q[:, i]

    return r, u


def solve_Q(Q, b):
    """
    Given a unitary mxm matrix Q and a vector b, solve Qx=b for x.

    :param Q: an mxm dimensional numpy array containing the unitary matrix
    :param b: the m dimensional array for the RHS

    :return x: m dimensional array containing the solution.
    """

    x = Q.conj().T @ b

    return x


def time_solve_Q(Q, b):
    start_time = time()
    solve_Q(Q, b)
    print(time() - start_time)

    start_time = time()
    np.linalg.solve(Q, b)
    print(time() - start_time)

# For a matrix of size 100, both took 0.0 seconds
# For a matrix of size 200,  np.linalg.solve was slightly slower than solve_Q
# For a matrix of size 400,  np.linalg.solve was again slightly slower than
# solve_Q


def orthog_proj(Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute the orthogonal projector P that projects vectors onto
    the subspace spanned by those vectors.

    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return P: an mxm-dimensional numpy array containing the projector
    """

    P = Q @ Q.conj().T

    return P


def orthog_space(V):
    """
    Given set of vectors u_1,u_2,..., u_n, compute the
    orthogonal complement to the subspace U spanned by the vectors.

    :param V: an mxn-dimensional numpy array whose columns are the \
    vectors u_1,u_2,...,u_n.

    :return Q: an mxl-dimensional numpy array whose columns are an \
    orthonormal basis for the subspace orthogonal to U, for appropriate l.
    """
    m, n = np.shape(V)
    Q, R = np.linalg.qr(V, "complete")
    P = Q @ Q.conj().T
    P_orth = np.eye(m) - P
    Q = (P_orth @ Q)[:, :m-n]

    return Q


def GS_classical(A):
    """
    Given an mxn matrix A, compute the QR factorisation by classical
    Gram-Schmidt algorithm, transforming A to Q in place and returning R.

    :param A: mxn numpy array

    :return R: nxn numpy array
    """

    n = np.size(A, 1)
    R = np.zeros((n, n), dtype=A.dtype)

    for j in range(n):
        R[:j, j] = A[:, :j].conj().T @ A[:, j]
        A[:, j] -= A[:, :j] @ R[:j, j]
        R[j, j] = np.linalg.norm(A[:, j])
        A[:, j] /= R[j, j]

    return R


def GS_modified(A):
    """
    Given an mxn matrix A, compute the QR factorisation by modified
    Gram-Schmidt algorithm, transforming A to Q in place and returning
    R.

    :param A: mxn numpy array

    :return R: nxn numpy array
    """

    n = np.size(A, 1)
    R = np.zeros((n, n), dtype=A.dtype)

    for i in range(n):
        R[i, i] = np.linalg.norm(A[:, i])
        A[:, i] /= R[i, i]
        R[i, i+1:] = A[:, i].conj().T @ A[:, i+1:]
        for j in range(i+1, n):
            A[:, j] -= R[i, j] * A[:, i]

    return R

# Ex 3.6


def mutual_orthogonality():
    A = np.random.rand(100, 100)
    A_copy = np.copy(A)
    R_classical = GS_classical(A)
    R_modified = GS_modified(A_copy)

    return A @ A_copy


def GS_modified_get_R(A, k):
    """
    Given an mxn matrix A, with columns of A[:, 0:k] assumed orthonormal,
    return upper triangular nxn matrix R such that
    Ahat = A*R has the properties that
    1) Ahat[:, 0:k] = A[:, 0:k],
    2) A[:, k] is normalised and orthogonal to the columns of A[:, 0:k].

    :param A: mxn numpy array
    :param k: integer indicating the column that R should orthogonalise

    :return R: nxn numpy array
    """

    n = np.size(A, 1)
    R = np.identity(n, dtype=A.dtype)

    R[k, k] = np.linalg.norm(A[:, k])
    R[k, k+1:] = A[:, k] @ A[:, k+1:]

    return R


def GS_modified_R(A):
    """
    Implement the modified Gram Schmidt algorithm using the lower triangular
    formulation with Rs provided from GS_modified_get_R.

    :param A: mxn numpy array

    :return Q: mxn numpy array
    :return R: nxn numpy array
    """

    m, n = A.shape
    A = 1.0*A
    R = np.eye(n, dtype=A.dtype)
    for i in range(n):
        Rk = GS_modified_get_R(A, i)
        A[:, :] = np.dot(A, Rk)
        R[:, :] = np.dot(R, Rk)
    R = np.linalg.inv(R)
    return A, R
