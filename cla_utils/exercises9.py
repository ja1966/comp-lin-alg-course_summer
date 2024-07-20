import numpy as np
import numpy.random as random
import cla_utils


def get_A100():
    """
    Return A100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    return A


def get_B100():
    """
    Return B100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A[np.tril_indices(m, -2)] = 0
    return A


def get_C100():
    """
    Return C100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = 0.5*(A + np.conj(A).T)
    return A


def get_D100():
    """
    Return D100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = 0.5*(A + np.conj(A).T)
    A[np.tril_indices(m, -2)] = 0
    A[np.triu_indices(m, 2)] = 0
    return A


def get_A3():
    """
    Return A3 matrix for investigating power iteration.

    :return A3: a 3x3 numpy array.
    """

    return np.array([[ 0.76505141, -0.03865876,  0.42107996],
                     [-0.03865876,  0.20264378, -0.02824925],
                     [ 0.42107996, -0.02824925,  0.23330481]])


def get_B3():
    """
    Return B3 matrix for investigating power iteration.

    :return B3: a 3x3 numpy array.
    """
    return np.array([[ 0.76861909,  0.01464606,  0.42118629],
                     [ 0.01464606,  0.99907192, -0.02666057],
                     [ 0.42118629, -0.02666057,  0.23330798]])


def pow_it(A, x0, tol, maxit, store_iterations = False):
    """
    For a matrix A, apply the power iteration algorithm with initial
    guess x0, until either

    ||r|| < tol where

    r = Ax - lambda*x,

    or the number of iterations exceeds maxit.

    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of power iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing all \
    the iterates.
    :return lambda0: the final eigenvalue.
    """

    nits = 0
    x = x0
    lambda0 = 0
    m = np.size(A, 0)

    if store_iterations:
        x_store = np.zeros((m, maxit))

    while nits <= maxit and np.linalg.norm(A@x - lambda0*x) >= tol:
        x = A@x
        x /= np.linalg.norm(x)
        lambda0 = x.T @ A @ x
        if store_iterations:
            x_store[nits] = x
        nits += 1

    if store_iterations:
        return x_store, lambda0

    return x, lambda0


def inverse_it(A, x0, mu, tol, maxit, store_iterations = False):
    """
    For a Hermitian matrix A, apply the inverse iteration algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.

    :param A: an mxm numpy array
    :param mu: a floating point number, the shift parameter
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of inverse iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing \
    all the iterates.
    :return l: a floating point number containing the final eigenvalue \
    estimate, or if store_iterations, a maxit dimensional numpy array containing \
    all the iterates.
    """

    nits = 0
    x = x0
    lambda0 = 0
    m = np.size(A, 0)

    if store_iterations:
        x_store = np.zeros((m, maxit))
        eig_store = np.zeros(maxit)

    while nits <= maxit and np.linalg.norm(A@x - lambda0*x) >= tol:
        w = cla_utils.householder_solve(A - mu * np.identity(m), x.reshape((m, 1)))
        x = w / np.linalg.norm(w)
        lambda0 = x.T @ A @ x

        if store_iterations:
            x_store[nits] = x
            eig_store[nits] = lambda0

        nits += 1

    if store_iterations:
        return x_store, eig_store

    return x, lambda0


def rq_it(A, x0, tol, maxit, store_iterations = False):
    """
    For a Hermitian matrix A, apply the Rayleigh quotient algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.

    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of inverse iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing \
    all the iterates.
    :return l: a floating point number containing the final eigenvalue \
    estimate, or if store_iterations, an m dimensional numpy array containing \
    all the iterates.
    """

    nits = 0
    x = x0
    lambda0 = x.T @ A @ x
    m = np.size(A, 0)

    if store_iterations:
        x_store = np.zeros((m, maxit))
        eig_store = np.zeros(maxit)

    while nits <= maxit and np.linalg.norm(A@x - lambda0*x) >= tol:
        w = cla_utils.householder_solve(A - lambda0 * np.identity(m), x.reshape((m, 1)))
        x = w / np.linalg.norm(w)
        lambda0 = x.T @ A @ x

        if store_iterations:
            x_store[nits] = x
            eig_store[nits] = lambda0

        nits += 1

    if store_iterations:
        return x_store, eig_store

    return x, lambda0


def pure_QR(A, maxit, tol):
    """
    For matrix A, apply the QR algorithm and return the result. Convergence is
    based on the sum of the entries below the diagonal being lower than the
    specified tolerance.

    :param A: an mxm numpy array
    :param maxit: the maximum number of iterations
    :param tol: termination tolerance

    :return Ak: the result
    """

    nits = 0
    m = np.size(A, 0)
    x, lambda_max = pow_it(A, np.ones(m), tol, maxit)
    Ak = A

    while nits <= maxit and np.abs(np.tril(Ak).sum()-np.trace(Ak)) >= tol:
        Q, R = cla_utils.householder_qr(Ak)
        Ak = R @ Q
        nits += 1

    return Ak
