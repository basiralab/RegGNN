'''
Tools for computing topological features in Riemannian space.

Code taken from https://morphomatics.github.io/,
created by Felix Ambellan and Martin Hanik and Christoph von Tycowicz, 2021.
'''

import numpy as np
import numpy.random as rnd
import numpy.linalg as la

from scipy.linalg import logm, expm_frechet

from pymanopt.manifolds.manifold import Manifold
from pymanopt.tools.multi import multisym


class SPD(Manifold):
    """Returns the product manifold Sym+(d)^k, i.e., a product of k dxd symmetric positive matrices (SPD).

     manifold = SPD(k, d)

     Elements of Sym+(d)^k are represented as arrays of size kxdxd where every dxd slice is an SPD matrix, i.e., a
     symmetric matrix S with positive eigenvalues.

     The Riemannian metric used is the product Log-Euclidean metric that is induced by the standard Euclidean trace
     metric; see
                    Arsigny, V., Fillard, P., Pennec, X., and Ayache., N.
                    Fast and simple computations on tensors with Log-Euclidean metrics.
     """

    def __init__(self, k=1, d=3):
        if d <= 0:
            raise RuntimeError("d must be an integer no less than 1.")

        if k == 1:
            self._name = 'Manifold of symmetric positive definite {d} x {d} matrices'.format(d=d, k=k)
        elif k > 1:
            self._name = 'Manifold of {k} symmetric positive definite {d} x {d} matrices (Sym^+({d}))^{k}'.format(d=d, k=k)
        else:
            raise RuntimeError("k must be an integer no less than 1.")

        self._k = k
        self._d = d

    def __str__(self):
        return self._name

    @property
    def dim(self):
        return int((self._d*(self._d+1)/2) * self._k)

    @property
    def typicaldist(self):
        # typical affine invariant distance
        return np.sqrt(self._k * 6)

    def inner(self, S, X, Y):
        """product metric"""
        return np.sum(np.einsum('...ij,...ij', X, Y))

    def norm(self, S, X):
        """norm from product metric"""
        return np.sqrt(self.inner(S, X, X))

    def proj(self, X, H):
        """orthogonal (with respect to the Euclidean inner product) projection of ambient
        vector ((k,3,3) array) onto the tangent space at X"""
        return dlog(X, multisym(H))

    def egrad2rgrad(self,X,D):
        # should be adj_dexp instead of dexp (however, dexp appears to be self-adjoint for symmetric matrices)
        return dexp(log_mat(X), multisym(D))

    def ehess2rhess(self, X, Hess):
        # TODO
        return

    def exp(self, S, X):
        """Riemannian exponential with base point S evaluated at X"""
        assert S.shape == X.shape

        # (avoid additional exp/log)
        Y = X + log_mat(S)
        vals, vecs = la.eigh(Y)
        return np.einsum('...ij,...j,...kj', vecs, np.exp(vals), vecs)

    retr = exp

    def log(self, S, U):
        """Riemannian logarithm with base point S evaluated at U"""
        assert S.shape == U.shape

        # (avoid additional log/exp)
        return log_mat(U) - log_mat(S)

    def geopoint(self, S, T, t):
        """ Evaluate the geodesic from S to T at time t in [0, 1]"""
        assert S.shape == T.shape and np.isscalar(t)

        return self.exp(S, t * self.log(S, T))

    def rand(self):
        S = np.random.random((self._k, self._d, self._d))
        return np.einsum('...ij,...kj', S, S)

    def randvec(self, X):
        Y = self.rand()
        y = self.log(X, Y)
        return y / self.norm(X, y)

    def zerovec(self, X):
        return np.zeros((self._k, self._d, self._d))

    def transp(self, S, T, X):
        """Parallel transport for Sym+(d)^k.
        :param S: element of Symp+(d)^k
        :param T: element of Symp+(d)^k
        :param X: tangent vector at S
        :return: parallel transport of X to the tangent space at T
        """
        assert S.shape == T.shape == X.shape

        # if X were not in algebra but at tangent space at S
        #return dexp(log_mat(T), dlog(S, X))

        return X

    def eleminner(self, R, X, Y):
        """element-wise inner product"""
        return np.einsum('...ij,...ij', X, Y)

    def elemnorm(self, R, X):
        """element-wise norm"""
        return np.sqrt(self.eleminner(R, X, X))

    def projToGeodesic(self, X, Y, P, max_iter=10):
        '''
        :arg X, Y: elements of Symp+(d)^k defining geodesic X->Y.
        :arg P: element of Symp+(d)^k to be projected to X->Y.
        :returns: projection of P to X->Y
        '''

        assert X.shape == Y.shape
        assert Y.shape == P.shape

        # all tagent vectors in common space i.e. algebra
        v = self.log(X, Y)
        v /= self.norm(X, v)

        w = self.log(X, P)
        d = self.inner(X, v, w)

        return self.exp(X, d * v)

    def pairmean(self, S, T):
        assert S.shape == T.shape

        return self.exp(S, 0.5 * self.log(S, T))

    def dist(self, S, T):
        """Distance function in Sym+(d)^k"""
        return self.norm(S, self.log(S,T))

    def adjJacobi(self, S, T, t, X):
        """Evaluates an adjoint Jacobi field along the geodesic gam from S to T
        :param S: element of the space of differential coordinates
        :param T: element of the space of differential coordinates
        :param t: scalar in [0,1]
        :param X: tangent vector at gam(t)
        :return: tangent vector at X
        """
        assert S.shape == T.shape == X.shape and np.isscalar(t)

        U = self.geopoint(S, T, t)
        return (1 - t) * self.transp(U, S, X)

    def adjDxgeo(self, S, T, t, X):
        """Evaluates the adjoint of the differential of the geodesic gamma from S to T w.r.t the starting point S at X,
        i.e, the adjoint  of d_S gamma(t; ., T) applied to X, which is en element of the tangent space at gamma(t).
        """
        assert S.shape == T.shape == X.shape and np.isscalar(t)

        return self.adjJacobi(S, T, t, X)

    def adjDygeo(self, S, T, t, X):
        """Evaluates the adjoint of the differential of the geodesic gamma from S to T w.r.t the endpoint T at X,
        i.e, the adjoint  of d_T gamma(t; S, .) applied to X, which is en element of the tangent space at gamma(t).
        """
        assert S.shape == T.shape == X.shape and np.isscalar(t)

        return self.adjJacobi(T, S, 1 - t, X)

def log_mat(U):
    """Matrix logarithm, only use for normal matrices U, i.e., U * U^T = U^T * U"""
    vals, vecs = la.eigh(U)
    vals = np.log(np.where(vals > 1e-10, vals, 1))
    return np.real(np.einsum('...ij,...j,...kj', vecs, vals, vecs))

def dexp(X, G):
    """Evaluate the derivative of the matrix exponential at
    X in direction G.
    """
    return np.array([expm_frechet(X[i],G[i])[1] for i in range(X.shape[0])])

def dlog(X, G):
    """Evaluate the derivative of the matrix logarithm at
    X in direction G.
    """
    n = X.shape[1]
    # set up [[X, G], [0, X]]
    W = np.hstack((np.dstack((X, G)), np.dstack((np.zeros_like(X), X))))
    return np.array([logm(W[i])[:n, n:] for i in range(X.shape[0])])


def vectime3d(x, A):
    """
    :param x: vector of length k
    :param A: array of size k x n x m
    :return: k x n x m array such that the j-th n x m slice of A is multiplied with the j-th element of x
    """
    assert np.size(x.shape[0]) == 2 and np.size(A) == 3
    assert x.shape[0] == 1 or x.shape[1] == 1
    assert x.shape[0] == A.shape[0] or x.shape[1] == A.shape[0]

    if x.shape[0] == 1:
        x = x.T
    A = np.einsum('kij->ijk', A)
    return np.einsum('ijk->kij', x * A)

def vectime3dB(x, A):
    """
    :param x: vector of length k
    :param A: array of size k x n x m
    :return: k x n x m array such that the j-th n x m slice of A is multiplied with the j-th element of x

    In case of k=1, x * A is returned.
    """
    if np.isscalar(x) and A.ndim == 2:
        return x * A

    x = np.atleast_2d(x)
    assert x.ndim <= 2 and np.size(A.shape) == 3
    assert x.shape[0] == 1 or x.shape[1] == 1
    assert x.shape[0] == A.shape[0] or x.shape[1] == A.shape[0]

    if x.shape[1] == 1:
        x = x.T
    A = np.einsum('kij->ijk', A)
    return np.einsum('ijk->kij', x * A)
