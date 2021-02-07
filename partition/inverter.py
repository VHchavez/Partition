"""
inverter.py
"""

import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt

from opt_einsum import contract

# from pyscf import scf

from .util import to_grid, basis_to_grid, get_from_grid

import psi4
psi4.core.be_quiet()

import sys

class Inverter():

    def __init__(self, partition_object,
                       debug=False):

        self.part     = partition_object
        self.inv_type = "partition"
        self.opt_method = None

        self.nauxbf   = self.part.nbf

        self.grad_a = None
        self.grad_b = None

        #Target Quantities
        self.ct = None
        self.nt = None

        #Inverted Potential
        self.v = np.zeros( 2 * self.nauxbf )
        self.reg = 0.0

        self.debug  = debug

    def initial_guess(self, guess):
        """
        Provides initial guess for inversion
        """ 

        if self.inv_type == "xc":

            print("Generating Initial Guess")

        elif self.inv_type == 'partition':

            #Let us try to do same guess. This guess will first be FA for the whole system. 
            #If this doesn't work I'll try sum of fermi amaldis. 
            N = self.part.mol.nalpha + self.part.mol.nbeta

            if guess.lower() == 'none':
                self.va = np.zeros_like(self.part.T)
                self.vb = np.zeros_like(self.part.T)

            if guess.lower() == 'fa':
                print("Calculating Fermi Amaldi")
                for f_i in self.part.frags:
                    N = f_i.nalpha + f_i.nbeta
                    coca = psi4.core.Matrix.from_array(f_i.Ca_occ)
                    cocb = psi4.core.Matrix.from_array(f_i.Cb_occ)
                    J, _ = self.part.form_jk(  coca , cocb )
                    fa =  (- 1.0 / N) * (J[0] + J[1])
                    self.va = fa
                    self.vb = fa

            elif guess.lower() == 'hfa':
                print("Calculating Hartree Fermi Amaldi")
                for f_i in self.part.frags:
                    N = f_i.nalpha + f_i.nbeta
                    coca = psi4.core.Matrix.from_array(f_i.Ca_occ)
                    cocb = psi4.core.Matrix.from_array(f_i.Cb_occ)
                    J, _ = self.part.form_jk(  coca , cocb )
                    fa =  (1 - 1.0 / N) * (J[0] + J[1])
                    self.va = fa
                    self.vb = fa


    #PDFT INVERSION ###################################################################################

    def pdft_invert(self, target_density, 
                     target_cocc=None, 
                     opt_method='trust-krylov', 
                     v_guess="core"):

        self.ct = target_cocc
        self.nt = target_density
        self.initial_guess(v_guess)
        self.opt_method = opt_method

        if self.debug == True:
            print("Optimizing")
    
        if self.opt_method.lower() == 'bfgs':
            opt_results = minimize( fun = self.pdft_lagrangian,
                                    x0  = self.v, 
                                    jac = self.pdft_gradient,
                                    method = self.opt_method,
                                    # tol    = 1e-10,
                                    options = {"maxiter" : 1000,
                                               "disp"    : False,}
                                    )

        else:
            opt_results = minimize( fun = self.pdft_lagrangian,
                                    x0  = self.v, 
                                    jac = self.pdft_gradient,
                                    hess = self.wy_hessian,
                                    method = self.opt_method,
                                    # tol    = 1e-10,
                                    options = {"maxiter" : 1000,
                                               "disp"    : False, }
                                    )

        # if opt_results.success is False:
        #     raise ValueError("Optimization was unsucessful, try a different intitial guess")
        
        print("Finalized Optimization Successfully")
        print(opt_results.message)

        self.opt_results = opt_results
        # self.pdft_finalize_energy()
        self.v = opt_results.x

    def pdft_lagrangian(self, v):
        vp_a = np.einsum("ijk,k->ij", self.part.S3, v[:self.part.nbf]) + self.va
        vp_b = np.einsum("ijk,k->ij", self.part.S3, v[self.part.nbf:]) + self.vb

        self.part.scf_frags(method="svwn", vext=(vp_a, vp_b))
        self.part.frag_sum()

        Da = self.part.frags_na
        Db = self.part.frags_nb

        self.grad_a = np.einsum('ij,ijt->t', (Da-self.nt[0]), self.part.S3)
        self.grad_b = np.einsum('ij,ijt->t', (Db-self.nt[1]), self.part.S3) 

        kinetic = 0.0
        potential = 0.0
        for i in range(self.part.nfrags):
            potential += self.part.frags[i].energies["total"]

        optz       = np.einsum('t,t', v[:self.nauxbf], self.grad_a)
        optz      += np.einsum('t,t', v[self.nauxbf:], self.grad_b)

        print(f" L1: {kinetic:4.10f}, L2: {potential:4.10f}, L3: {optz:4.10f}, v_norm: {np.linalg.norm(v):4.10f}, total_L: { kinetic + potential + optz}")

        L = kinetic + potential + optz

        penalty = 0.0
        # if self.reg > 0:
        #     penalty = self.reg * self.Dvb(v)

        return -(L - penalty)

    def pdft_gradient(self, v):

        vp_a = np.einsum("ijk,k->ij", self.part.S3, v[:self.part.nbf]) + self.va
        vp_b = np.einsum("ijk,k->ij", self.part.S3, v[self.part.nbf:]) + self.vb

        self.part.scf_frags(method="svwn", vext=[vp_a, vp_b])
        self.part.frag_sum()

        Da = self.part.frags_na
        Db = self.part.frags_nb

        self.grad_a = contract('ij,ijt->t', (Da-self.nt[0]), self.part.S3)
        self.grad_b = contract('ij,ijt->t', (Db-self.nt[1]), self.part.S3) 

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

        vp_a = np.einsum("ijk,k->ij", self.part.S3, v[:self.part.nbf]) + self.va
        vp_b = np.einsum("ijk,k->ij", self.part.S3, v[self.part.nbf:]) + self.vb
        self.part.scf_frags(method="svwn", vext=[vp_a, vp_b])

        Hs = np.zeros((2 * self.part.nbf, 2 * self.part.nbf))
        for fi in self.part.frags:
            na, nb = fi.nalpha, fi.nbeta
            eigs_diff_a = fi.eig_a[:na, None] - fi.eig_a[None, na:]
            C3a = contract('mi,va,mvt->iat', fi.Ca[:,:na], fi.Ca[:,na:], self.part.S3)
            Ha = 2 * contract('iau,iat,ia->ut', C3a, C3a, eigs_diff_a**-1)

            Hs += np.block(
                            [[Ha,                                      np.zeros((self.part.nbf, self.part.nbf))],
                            [np.zeros((self.part.nbf, self.part.nbf)), Ha                              ]]
                        )
        return - Hs

    def pdft_finalize_energy(self):

        pass


        