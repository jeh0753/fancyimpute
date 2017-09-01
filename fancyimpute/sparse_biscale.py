import numpy as np 
#from suvc import suvc # this is a fortran package from suvC.f - built with line 'f2py -c -m suvc suvC.f'
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
#class Incomplete(csr_matrix):
    

class BiScale(object):
    def __init__(self, x, maxit=20, thresh=1e-9, row_center=True, row_scale=False, col_center=True, col_scale=False, trace=False):
        self.x = x
        self.maxit = maxit
        self.thresh = 1e-9
        self.row_center = row_center
        self.row_scale = row_scale
        self.col_center = col_center
        self.col_scale = col_scale
        self.trace = trace
        self.m = x.shape[0]
        self.n = x.shape[1]
        self.a = np.zeros(self.m)
        self.b = np.zeros(self.n)
        self.tau = np.ones(self.m)
        self.gamma = np.ones(self.n)
        self.xhat = self.x.copy()
        self.critmat = []
    
    def _prepare_suvc(self):
        a = self.a.copy()
        a = a.reshape(-1,1)
        b = self.b.copy()
        b = b.reshape(-1,1)
        a = np.hstack((a, np.ones(a.shape[0]).reshape(-1,1)))
        b = np.hstack((np.ones(b.shape[0]).reshape(-1,1), b))
        return a, b 
    
    def _pred_one(self, u, v, row, col):
        u_data = np.expand_dims(u[row,:], 0)
        return float(u_data.dot(v[col, :].T))

    def _c_suvc(self, u, v, irow, icol):
        #nnrow = np.int32(u.shape[0])
        #nncol = np.int32(v.shape[0])
        #nrank = np.int32(u.shape[1]) 
        nomega = len(irow)
        #u = np.require(u, np.float32, requirements = ['A','O','W','F'])
        #v = np.require(v, np.float32, requirements = ['A','O','W','F'])
        res = np.zeros(nomega)
        targets = zip(irow, icol)
        for idx, (r,c) in enumerate(targets):
            res[idx] = self._pred_one(u, v, r, c)
        #res = suvc(nnrow, nncol, nrank, y, z, irow, pcol, nomega, res)
        #res = suvc(res, nomega, pcol, irow, v, u, nrank, nncol, nnrow)
        return res

    def _center_scale_I(self): 
        x = self.x.data
        #suvc = self._convert_for_suvc()
        a, b = self._prepare_suvc()
        coo_x = coo_matrix(self.x)
        irow = coo_x.row
        icol = coo_x.col
        #import ipdb; ipdb.set_trace()
        suvc1 = self._c_suvc(a, b, irow, icol)  
        suvc2 = self._c_suvc(self.tau.reshape(-1,1), self.tau.reshape(-1,1), irow, icol)
        self.xhat.data = (x-suvc1) / suvc2
        #b_center = suvc(nnrow, nncol,nrank, self.tau, self.gamma, self.x.indices, self.x.indptr, nomega)  
        #nnrow,nncol,nrank,u,v,irow,pcol,nomega
        return self
        
    def _col_sum_along(self, a, x):
        x = (self.x != 0)
        a = csc_matrix(a.T)
        return a.dot(x).toarray()

    def _row_sum_along(self, b, x):
        #TODO - still make b into a csc?
        x = (self.x != 0)
        return x.dot(b)

    def solve(self):
        self._center_scale_I()
        for i in xrange(self.maxit):
            # Centering
            ## Column mean
            if self.col_center:
                colsums = np.sum(self.xhat, axis=0)
                gamma_by_sum = np.multiply(colsums,(self.gamma))
                dbeta = gamma_by_sum / self._col_sum_along(1 / self.tau, self.x)
                self.b = self.b + dbeta
                self._center_scale_I()
            else:
                dbeta = 0
            
            ## Row Mean
            if self.row_center:
                rowsums = np.sum(self.xhat, axis=1).T
                tau_by_sum = np.multiply(self.tau, rowsums)
                dalpha = tau_by_sum / self._row_sum_along(1 / self.gamma, self.x)
                self.a = self.a + dalpha
                self._center_scale_I()

            else:
                dalpha = 0 
        
            #Leaving out scaling for now; does not appear to be used for my purposes
            convergence_level = np.square(dalpha).sum() + np.square(dbeta).sum()
            self.critmat.append([i + 1, convergence_level])
            if convergence_level < self.thresh:
                break

        # Complete solution
        self.xhat.row_center = np.ravel(self.a)
        self.xhat.col_center = np.ravel(self.b)
        self.xhat.row_scale = np.ravel(self.tau)
        self.xhat.col_scale = np.ravel(self.gamma)
        self.xhat.critmat = self.critmat

        return self.xhat




#    def _check_row_centering(self):
#        if isinstance(self.row_center, (int, long, float, complex)):
#            if len(self.row_center) == self.m:
#                self.a = self.row_center
#                self.row_center = False
#            else:
#                raise ValueError("length of row_center must equal the number of rows of 'x'")
#        else:
#            self.a = np.zeros(self.m)
#       
#    def _check_col_centering(self):
#        if isinstance(self.col_center, (int, long, float, complex)):
#            if len(self.col_center) == self.n:
#                self.b = self.col_center
#                self.col_center = False
#            else:
#                raise ValueError("length of col_center must equal the number of columns of 'x'")
#        else:
#            self.b = np.zeros(self.n)

    
