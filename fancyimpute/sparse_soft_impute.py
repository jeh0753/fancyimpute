# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, print_function, division

from six.moves import range
import numpy as np
from numpy.random import RandomState
from sklearn.utils.extmath import randomized_svd
from scipy.sparse import coo_matrix, csc_matrix, issparse

from biscaler import BiScaler
#from .common import masked_mae
from sparse_solver import Solver

class SPLR(object):
    
    def __init__(self, x, a=None, b=None):
        self.x = x
        self.a = a
        self.b = b

        x_dims = x.shape

        if a is None:
            self.b = None

        if b is None:
            self.a = None
        else:
            a_dims = a.shape
            b_dims = b.shape
            if a_dims[0] != x_dims[0]:
                raise ValueError("number of rows of x not equal to number of rows of a")

            if b_dims[0] != x_dims[1]:
                raise ValueError("number of columns of x not equal to number of rows of b")

            if a_dims[1] != b_dims[1]:
                raise ValueError("number of columns of a not equal to number of columns of b")

    def r_mult(self, other):
        """Left Multiplication
        This is equivalent to self.dot(other)
        """
        result = self.x.dot(other)
        result = result

        if self.a is not None:
            b_mult = self.b.T.dot(other)
            ab_mult = self.a.dot(b_mult)
            result += ab_mult

        return result

    def l_mult(self, other):
        """Left Multiplication
        This is equivalent to other.dot(self)
        """
        result = csc_matrix(other).dot(self.x) # conversion necessary for dot to be called successfully
        result = result.toarray()

        if self.a is not None:
            ab_mult = other.dot(self.a)
            ab_mult = ab_mult.dot(self.b.T)
            result += ab_mult

        return result
    

class SoftImpute(Solver):
    """
    Implementation of the SoftImpute algorithm from:
    "Spectral Regularization Algorithms for Learning Large Incomplete Matrices"
    by Mazumder, Hastie, and Tibshirani.
    """
    def __init__(
            self,
            shrinkage_value=0,
            convergence_threshold=1e-05,
            max_iters=100,
            max_rank=2, # TODO: add logic so this cant be > m
            n_power_iterations=8,
            init_fill_method="zero",
            min_value=None,
            max_value=None,
            normalizer=None,
            verbose=True):
        """
        Parameters
        ----------
        shrinkage_value : float
            Value by which we shrink singular values on each iteration. If
            omitted then the default value will be the maximum singular
            value of the initialized matrix (zeros for missing values) divided
            by 100.

        convergence_threshold : float
            Minimum ration difference between iterations (as a fraction of
            the Frobenius norm of the current solution) before stopping.

        max_iters : int
            Maximum number of SVD iterations

        max_rank : int, optional
            Perform a truncated SVD on each iteration with this value as its
            rank.

        n_power_iterations : int
            Number of power iterations to perform with randomized SVD

        init_fill_method : str
            How to initialize missing values of data matrix, default is
            to fill them with zeros.

        min_value : float
            Smallest allowable value in the solution

        max_value : float
            Largest allowable value in the solution

        normalizer : object
            Any object (such as BiScaler) with fit() and transform() methods

        verbose : bool
            Print debugging info
        """
        Solver.__init__(
            self,
            fill_method=init_fill_method,
            min_value=min_value,
            max_value=max_value,
            max_rank=max_rank,
            normalizer=normalizer)
        self.shrinkage_value = shrinkage_value
        self.convergence_threshold = convergence_threshold
        self.max_iters = max_iters
        self.max_rank = max_rank
        self.n_power_iterations = n_power_iterations
        self.verbose = verbose
        self.m = None
        self.n = None

#    def _converged(self, X_old, X_new, missing_mask):
#        # check for convergence
#        old_missing_values = X_old[missing_mask]
#        new_missing_values = X_new[missing_mask]
#        difference = old_missing_values - new_missing_values
#        ssd = np.sum(difference ** 2)
#        old_norm = np.sqrt((old_missing_values ** 2).sum())
#        return (np.sqrt(ssd) / old_norm) < self.convergence_threshold

    def _fnorm(self, SVD_old, SVD_new):
        # U, S, V is the order of SVD matrices. This function takes the Frobenius Norm of an SVD decomposed matrix.
        # TODO - make sure this equation is correct. It is borrowed from travisbrady's py-soft-impute package
        #import ipdb; ipdb.set_trace()
        U_old, D_sq_old, V_old = SVD_old
        U_new, D_sq_new, V_new = SVD_new
        utu = D_sq_new.dot(U_new.T.dot(U_old))
        vtv = D_sq_old.dot(V_old.T.dot(V_new))
        uvprod = utu.dot(vtv).sum()
        sing_val_sumsq_old = (D_sq_old ** 2).sum()
        sing_val_sumsq_new = (D_sq_new ** 2).sum()
        norm = (sing_val_sumsq_old + sing_val_sumsq_new - (2 * uvprod)) / max(sing_val_sumsq_old, 1e-9) 
        return norm

    def _converged(self, SVD_old, SVD_new):
        # U, S, V is the order of SVD matrices. This function takes the Frobenius Norm of an SVD decomposed matrix.
        norm = self._fnorm(SVD_old, SVD_new)
        return norm < self.convergence_threshold

    def _UD(self, U, D, n):
        ones = np.ones(n)
        return U * np.outer(ones, D)

    def _svd(self, X, max_rank=None):
        if issparse(X): # this will raise an error if integers are sent, rather than floats. np.array([float(i) for i in X.data]) is one solution, if we need it
            if max_rank is None:
                max_rank = min(X.shape)
            return linalg.svds(X, k=max_rank, maxiter=self.n_power_iterations) 

        elif max_rank:
            # if we have a max rank then perform the faster randomized SVD
            return randomized_svd(
                X,
                max_rank,
                n_iter=self.n_power_iterations)
        else:
            # perform a full rank SVD using ARPACK
            return np.linalg.svd(
                X,
                full_matrices=False,
                compute_uv=True)

    def _x_real_pred(self, U, s, V, X_sparse, missing_mask):
        """predicts X values only for specified indices/columns"""
        nnz = X_sparse.nnz
        res = np.zeros(nnz)
        row_id, col_id, _ = missing_mask # row and column indices are drawn from missing_mask to avoid unnecessary conversion to COO format
        targets = zip(row_id, col_id)
        for idx, (r, c) in enumerate(targets):
            #import ipdb; ipdb.set_trace()
            res[idx] = np.sum(U[r]*s*V[:,c])
            #print(res[idx])
        return res

    def _svd_step(self, X, shrinkage_value):
        """
        Returns reconstructed X from low-rank thresholded SVD and
        the rank achieved.
        """
        U, s, V = self._svd(X, self.max_rank)
        s_thresh = np.maximum(s - shrinkage_value, 0)
        rank = (s_thresh > 0).sum()
        s_thresh = s_thresh[:rank]
        U_thresh = U[:, :rank]
        V_thresh = V[:rank, :]
        s_thresh = np.diag(s_thresh)
        X_reconstruction = np.dot(U_thresh, np.dot(S_thresh, V_thresh))
        return X_reconstruction, rank

    def _max_singular_value(self, X_filled):
        # quick decomposition of X_filled into rank-1 SVD
        if self.fill_method == 'sparse':
            return X_filled[1][0]
        else:
            _, s, _ = randomized_svd(
                X_filled,
                1,
                n_iter=5)
            return s[0]

    def _als_step(self, X_fill_svd, X_prev):
        U, D_sq, V = X_fill_svd
        U_old, D_sq_old, V_old = U.copy(), D_sq.copy(), V.copy()

        # U Step
       # import ipdb; ipdb.set_trace()
        B = (X_prev.l_mult(U.T)).T
        if self.shrinkage_value > 0:
            B = self._UD(B, D_sq / (D_sq + self.shrinkage_value), self.m)
        V, D_sq, _ = self._svd(B) # V is set to the U slot from V's SVD on purpose

        # V Step
        #obj=(.5*Frobsmlr(x,U,B,nx=normx)^2+lambda*sum(Dsq))/nz
        A = X_prev.r_mult(V)
        if self.shrinkage_value > 0:
            A = self._UD(A, D_sq / (D_sq + self.shrinkage_value), self.n)
        U, D_sq, V_part = self._svd(A)
        V = V.dot(V_part.T) # just for computing the convergence criterion

        ratio = self._fnorm((U_old,D_sq_old,V_old),(U,D_sq,V))
        converged = (ratio < self.convergence_threshold)
        X_fill_svd = (U, D_sq, V)
        return X_fill_svd, converged
    
    def _als(self, X_fill_svd, X_prev):
        for i in range(self.max_iters):
            X_fill_svd, converged = self._als_step(X_fill_svd, X_prev) 
            if converged:
                break
        if self.shrinkage_value > 0:
            U, D_sq, V = X_fill_svd
            A = X_prev.r_mult(V)
            U, D_sq, V_part = self._svd(A) 
            V = V.dot(V_part.T)
            D_sq = np.clip(D_sq - self.shrinkage_value, a_min=0, a_max=None) # this shrinks the singular values by lambda and clips them at zero
            X_fill_svd = (U, D_sq, V) #TODO - make this a single V_step function, since its used in _als_step as well
        return X_fill_svd

    def solve(self, X, missing_mask, X_original=None, max_rank=None):
        """
        X : 3-d or 2-d array
            X is a simple 2-d array if the input is dense. 
            X is a 3-d array composed of a U matrix, a D singular values array, and a V matrix if the input is sparse.

        missing_mask : array or list of matrices
            missing_mask is [x_true_rows, x_true cols, x_dense_shape] if the matrix is sparse. Otherwise, missing_mask is a boolean array indicating if value is nan at any given index of the original X matrix. 
        """
        X_filled = X
        if self.fill_method != 'sparse':
            observed_mask = ~missing_mask

        max_singular_value = self._max_singular_value(X_filled)
        J = max_rank

        if X_original is not None:
            self.n, self.m = X_original.shape
            x_res = X_original.copy()
            X_fill_svd = X_filled # renaming because X_filled is an SVD, if the original matrix was sparse. Its clunky do do it this way, but if X_original was passed it is safe to say we had a sparse matrix to start.

        if self.verbose:
            print("[SoftImpute] Max Singular Value of X_init = %f" % (
                max_singular_value))

        if self.shrinkage_value:
            shrinkage_value = self.shrinkage_value
        else:
            # totally hackish heuristic: keep only components
            # with at least 1/50th the max singular value
            shrinkage_value = max_singular_value / 50.0

        for i in range(self.max_iters):
            if self.fill_method != 'sparse':
                X_reconstruction, rank = self._svd_step(X_filled, shrinkage_value)
                converged = self._converged(self._svd(X_filled), self._svd(X_reconstruction))
                X_filled[missing_mask] = X_reconstruction[missing_mask]
#                max_rank=self.max_rank)
#            X_reconstruction = self.clip(X_reconstruction)

            else:
                X_fill_svd_old = X_fill_svd
                #import ipdb; ipdb.set_trace()
                U, Dsq, V = X_fill_svd

                if i == 0:
                    X_filled = SPLR(x_res)

                else:
                    BD = self._UD(V, Dsq, self.m)
                    x_hat = self._x_real_pred(U, Dsq, V.T, X_original, missing_mask)
                    x_res.data = X_original.data - x_hat # TODO - this may not work if .data isn't a type for the input data source. Also, can the input source data be overwritten like this?
                    X_filled = SPLR(x_res, U, BD)

                X_fill_svd = self._als(X_fill_svd, X_filled)
                converged = self._converged(X_fill_svd_old, X_fill_svd)

            # print error on observed data
#            if self.verbose:
#                mae = masked_mae(
#                    X_true=X_init,
#                    X_pred=X_reconstruction,
#                    mask=observed_mask)
#                print(
#                    "[SoftImpute] Iter %d: observed MAE=%0.6f rank=%d" % (
#                        i + 1,
#                        mae,
#                        rank))

#            converged = self._converged(
#                X_old=X_filled,
#                X_new=X_reconstruction,
#                missing_mask=missing_mask)
#            X_filled[missing_mask] = X_reconstruction[missing_mask]
            if converged:
                break

        if self.verbose:
            print("[SoftImpute] Stopped after iteration %d for lambda=%f" % (
                i + 1,
                shrinkage_value))

        if self.fill_method != 'sparse':
            return X_filled

        else:
            U, D_sq, V = X_fill_svd
            A = X_filled.r_mult(V)
            U, D_sq, V_part = self._svd(A)
            V = X_filled.a.dot(V_part.T)
            D_sq = np.clip(D_sq - self.shrinkage_value, a_min=0, a_max=None)
            J = min((D_sq>0).sum(), J)
            x_fill_svd = U[:,:J], D_sq[:J], V[:, :J]
            return X_fill_svd


if __name__ == '__main__':
    # original version - mat = np.array([[0.8654889, 0.01565179, 0.1747903, 0, 0],[-0.6004172, 0, -0.2119090, 0, 0],[-0.7169292, 0, 0, 0.06437356, -0.09754133],[0.6965558, -0.50331812, 0.5584839, 1.54375663, 0],[1.2311610, -0.34232368, -0.8102688, -0.82006429, -0.13256942],[0.2664415, 0.14486388, 0, 0, -2.24087863]])
    # bscale = BiScaler(scale_rows=False,scale_columns=False)
    # xsc = bscale.fit_transform(mat)
    xsc = np.array([[ 0.1390011, -0.09270246, -0.04629866, 0, 0], [-0.4469535, 0,  0.44695354, 0, 0], [-0.8252855, 0, 0,  0.1160078,  0.7092777], [-0.1981945, -0.7993484,  0.16913247,  0.8089969, 0], [ 0.9662344,  0.01088327, -0.56979652, -0.9250004,  0.5176792], [ 0.3651963,  0.86175224, 0, 0, -1.2269486]])
    x_orig = csc_matrix(xsc)
    sf = SoftImpute(max_rank=3, shrinkage_value=1, init_fill_method='sparse')
   # X_original, missing_mask = sf.prepare_input_data(x_orig)
   # X_svd = sf.fill(X_original, missing_mask, inplace=True)
   # U, Dsq, V = X_svd
   # m, n = X_original.shape
   # BD = sf._UD(V, Dsq, m) 
   # x_hat = sf._x_real_pred(U, Dsq, V, X_original.row, X_original.col)
   # x_res = X_original.copy()
   # x_res.data = X_original.data - x_hat
   # X_original = SPLR(x_res, U, BD)
   # sf._als_step(x_svd, x_original)
#    import ipdb; ipdb.set_trace()
    sf.single_imputation(x_orig)



