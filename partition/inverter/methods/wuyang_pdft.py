"""
wuyang.py
Handles wuyang type of inversion
"""

import numpy as np
from opt_einsum import contract
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import psi4


class WuYang_PDFT():
    """
    Performs Optimization as in: 10.1063/1.1535422 - Qin Wu + Weitao Yang
    Need to update to references on Nafziger 2014
    """

    def wuyang_pdft(self, 
                      opt_method='trust-krylov', 
                      initial_guess="none"):

        if self.optInv.verbose == True:
            print(f"Optimizing with method {opt_method}")
    
        if self.opt_method.lower() == 'bfgs':
            opt_results = minimize( fun = self.pdft_lagrangian,
                                    x0  = self.v0, 
                                    jac = self.pdft_gradient,
                                    method = self.opt_method,
                                    tol    = 1e-6,
                                    options = {"maxiter" : self.maxiter,
                                               "disp"    : False,}
                                    )

        else:
            opt_results = minimize( fun = self.pdft_lagrangian,
                                    x0  = self.v0, 
                                    jac = self.pdft_gradient,
                                    hess = self.wy_hessian,
                                    method = self.opt_method,
                                    tol    = 1e-6,
                                    options = {"maxiter" : self.maxiter,
                                               "disp"    : False, }
                                    )

        # if opt_results.success is False:
        #     raise ValueError("Optimization was unsucessful, try a different intitial guess")
        
        print(opt_results.message)

        self.opt_results = opt_results
        # self.pdft_finalize_energy()
        if opt_results.success:
            self.v = opt_results.x

        self.v = opt_results.x
        # else:
        #     self.v = self.vcurrent

    def diagonalize(self, matrix, ndocc):
        A = self.A
        Fp = A.dot(matrix).dot(A)
        eigvecs, Cp = np.linalg.eigh(Fp)
        C = A.dot(Cp)
        Cocc = C[:, :ndocc]
        D = contract('pi,qi->pq', Cocc, Cocc)
        return C, Cocc, D, eigvecs

    def pdft_lagrangian(self, v):

        va = np.einsum("ijk,k->ij", self.S3, v[:self.nbf])
        vb = np.einsum("ijk,k->ij", self.S3, v[self.nbf:])

        #Invert single molecular problem
        # if False:
        #     # Do a simple scf calculation. 
        #     _, _, da, _ = self.diagonalize(self.T + self.V + va, self.nalpha )
        #     _, _, db, _ = self.diagonalize(self.T + self.V + vb, self.nbeta  )       

        #     # This has to go in both versions
        #     self.grad_a = np.einsum('ij,ijt->t', (da-self.dt[0]), self.S3)
        #     self.grad_b = np.einsum('ij,ijt->t', (db-self.dt[1]), self.S3) 
        #     optz       = np.einsum('t,t', v[:self.nauxbf], self.grad_a)
        #     optz      += np.einsum('t,t', v[self.nauxbf:], self.grad_b)
        #     grad_value = np.max(self.grad_a + self.grad_b)

        #     # Calculate Energy components
        #     kinetic    =     np.sum( self.T * da )
        #     potential  =     np.sum( (self.V + self.va) * (da - self.dt[0]) )

        #     grad_value = np.max( self.grad_a + self.grad_b )
        #     print(f"Kinetic: {kinetic:6.10f} + Potential:{potential:6.10f} | Optimization: {optz:6.10f} | Gradient {grad_value:6.10f}" )
           
        #     if self.ref == 1:
        #         L = kinetic * potential * optz
        #     else:
        #         print("NO")
           
        
        #Invert as fragment molecules
        if True:
            da = np.zeros_like(self.dt[0])
            db = np.zeros_like(self.dt[0])

            self.current_vp = va

            for ifrag in self.frags:
                ifrag.scf(vext=[va, vb])
                da += ifrag.da.copy()
                db += ifrag.db.copy()

            # This has to go in both versions
            self.grad_a = np.einsum('ij,ijt->t', (da-self.dt[0]), self.S3)
            self.grad_b = np.einsum('ij,ijt->t', (db-self.dt[1]), self.S3) 
            grad_value = np.max(self.grad_a + self.grad_b)

            frag_energies = 0.0
            for i in range(self.nfrags):
                frag_energies += self.frags[i].E.Etot

            # This has to go in both versions
            optz       = np.einsum('t,t', v[:self.nauxbf], self.grad_a)
            optz      += np.einsum('t,t', v[self.nauxbf:], self.grad_b)

            print(f" Grad: {grad_value:4.10f}, Frags E: {frag_energies:4.10f}, Optimization: {optz:4.10f}, total_L: { frag_energies + optz}")

            L = frag_energies + optz

        penalty = 0.0
        # if self.reg > 0:
        #     penalty = self.reg * self.Dvb(v)

        return -(L - penalty)

    def pdft_gradient(self, v):
    
        va = np.einsum("ijk,k->ij", self.S3, v[:self.nbf])
        vb = np.einsum("ijk,k->ij", self.S3, v[self.nbf:])
        # _, _, da, _ = self.diagonalize(self.T + self.V + va, self.nalpha )
        # _, _, db, _ = self.diagonalize(self.T + self.V + vb, self.nbeta  )     

        # self.grad_a = np.einsum('ij,ijt->t', (da-self.dt[0]), self.S3)
        # self.grad_b = np.einsum('ij,ijt->t', (db-self.dt[1]), self.S3)
        da = np.zeros_like(self.dt[0])
        db = np.zeros_like(self.dt[0]) 

        for ifrag in self.frags:
            ifrag.scf(vext=[va, vb])
            da += ifrag.da.copy()
            db += ifrag.db.copy()

        self.grad_a = contract('ij,ijt->t', (da-self.dt[0]), self.S3)
        self.grad_b = contract('ij,ijt->t', (db-self.dt[1]), self.S3) 

        if self.reg > 0:
            self.grad_a = 2 * self.reg * contract('st,t->s', self.part.T, v[:self.nauxbf])
            self.grad_b = 2 * self.reg * contract('st,t->s', self.part.T, v[self.nauxbf:])

        self.grad = np.concatenate( (self.grad_a, self.grad_b) )

        return -self.grad

    def wy_hessian(self, v):
        """
        Calculates gradient wrt target density
        Equation (13) of main reference
        """
        # va = np.einsum("ijk,k->ij", self.S3, v[:self.nbf])
        # vb = np.einsum("ijk,k->ij", self.S3, v[self.nbf:])
        # ca, _, da, eigs_a = self.diagonalize(self.T + self.V + va, self.nalpha )
        # cb, _, db, eigs_b = self.diagonalize(self.T + self.V + vb, self.nbeta  )   

        # Hs = np.zeros((1 * self.nbf, 1 * self.nbf))
        # na, nb = self.nalpha, self.nbeta
        # eigs_diff_a = eigs_a[:na, None] - eigs_a[None, na:]
        # C3a = contract('mi,va,mvt->iat', ca[:,:na], ca[:,na:], self.S3)
        # Ha = 2 * contract('iau,iat,ia->ut', C3a, C3a, eigs_diff_a**-1)

        # Hs = np.block(
        #                 [[Ha,                              np.zeros((self.nbf, self.nbf))],
        #                 [np.zeros((self.nbf, self.nbf)), Ha                              ]]
        #              )

        vp_a = np.einsum("ijk,k->ij", self.part.S3, v[:self.part.nbf]) + self.va
        vp_b = np.einsum("ijk,k->ij", self.part.S3, v[self.part.nbf:]) + self.vb
        self.part.scf_frags(vext=[vp_a, vp_b])

        Hs = np.zeros((1 * self.nbf, 1 * self.nbf))
        for ifrag in self.frags:
            na, nb = ifrag.nalpha, ifrag.nbeta
            eigs_diff_a = ifrag.eigs_a[:na, None] - ifrag.eigs_a[None, na:]
            C3a = contract('mi,va,mvt->iat', ifrag.ca[:,:na], ifrag.ca[:,na:], self.S3)
            Ha = 2 * contract('iau,iat,ia->ut', C3a, C3a, eigs_diff_a**-1)

            Hs += Ha

        Hs = np.block(
                        [[Ha,                              np.zeros((self.nbf, self.nbf))],
                        [np.zeros((self.nbf, self.nbf)), Ha                              ]]
                    )


        return - Hs

        