"""
inverter.py
"""

import numpy as np
from scipy.optimize import minimize

class Inverter():

    def __init__(self, partition_object, 
                       method='bfgs', 
                       guess="fermi_amaldi"):

        self.part     = partition_object
        self.inv_type = "xc" if self.part.nfrags == 1 else "partition"
        self.method = method

        self.nauxbf   = self.part.nbf

        self.grad_a = None
        self.grad_b = None
        
        self.v0 = np.zeros( 2 * self.nauxbf )

        self.reg = 1e-16

        #Initialize
        if self.part.frags is None:
            self.part.scf(method="hf")
        self.initial_guess(guess=guess)

    
    def initial_guess(self, guess):
        """
        Provides initial guess for inversion
        """ 

        if self.inv_type == "xc":

            if guess == "fermi_amaldi":

                N = self.part.frags[0].natoms 

                if self.part.frags[0].Ca is None or self.part.frags[0].Cb is None:
                    raise ValueError("No Molecular Orbital Coefficient has been found, run a scf calculation")

                Cocca = self.part.frags[0].Ca_occ
                Coccb = self.part.frags[0].Cb_occ

                J, _ = self.part.form_jk( Cocca, Coccb )
                v_FA = - 1.0 * (1/N) * (J[0] + J[1])

                guess_matrix = v_FA
            
            if guess == "core":
                guess_matrix =   self.part.T = self.part.V

            self.guess = guess_matrix

        else:
            raise ValueError("I am unable to generate the requested guess")



    def gradient(self, v):
        #Generate Kohn Sham Potential
        Vks_a = np.zeros( (self.part.nbf, self.part.nbf) )
        Vks_b = np.zeros( (self.part.nbf, self.part.nbf) )

        #Add the transformed v matrix to ao basis + initial guess
        #PDFT way
        va_ao = np.einsum("ijk,k->ij", self.part.S3, v[:self.part.nbf])
        vb_ao = np.einsum("ijk,k->ij", self.part.S3, v[self.part.nbf:])
        # print(va_ao)
        #KSPIES way 
        # va_ao = np.einsu('t,ijt->ij', v[:self.part.nbf], self.part.o3 )
        # va_ao = np.einsu('t,ijt->ij', v[self.part.nbf:], self.part.o3 )
        # print(va_ao)

        Vks_a += va_ao + self.guess
        Vks_b += vb_ao + self.guess

        #Feed inverted Kohn-Sham Potential
        #This replaces all of frags information
        self.part.scf( method="core_only", vext=[Vks_a, Vks_b], evaluate=True)

        dd_a = self.n_target[0] - self.part.frags[0].Da 
        dd_b = self.n_target[1] - self.part.frags[0].Db


        #Generate Gradient
        grad_a = np.einsum(  "ij,ijk->k", dd_a, self.part.S3  )
        grad_b = np.einsum(  "ij,ijk->k", dd_a, self.part.S3  )
        self.grad_a = grad_a
        self.grad_b = grad_b
        grad   = np.concatenate( (grad_a, grad_b) )

        return grad


    def lagrangian(self, v):
        #Generate Kohn Sham Potential
        Vks_a = np.zeros( (self.part.nbf, self.part.nbf) )
        Vks_b = np.zeros( (self.part.nbf, self.part.nbf) )

        if self.grad_a is None and self.grad_b is None:
            dd_a = self.n_target[0] - self.part.frags[0].Da 
            dd_b = self.n_target[1] - self.part.frags[0].Db
            self.grad_a = np.einsum(  "ij,ijk->k", dd_a, self.part.S3  )
            self.grad_b = np.einsum(  "ij,ijk->k", dd_a, self.part.S3  )

        #Add the transformed v matrix to ao basis + initial guess
        #PDFT way
        va_ao = np.einsum("ijk,k->ij", self.part.S3, v[:self.nauxbf])
        vb_ao = np.einsum("ijk,k->ij", self.part.S3, v[self.nauxbf:])
        #KSPIES way 
        # va_ao = np.einsu('t,ijt->ij', v[:self.part.nbf], self.part.o3 )
        # va_ao = np.einsu('t,ijt->ij', v[self.part.nbf:], self.part.o3 )
        # print(va_ao)

        Vks_a += va_ao + self.guess
        Vks_b += vb_ao + self.guess

        #Feed inverted Kohn-Sham Potential
        #This replaces all of frags information
        self.part.scf( method="core_only", vext=[Vks_a, Vks_b], evaluate=True)

        na = self.part.frags[0].Da
        nb = self.part.frags[0].Db
        na_target = self.n_target[0]
        nb_target = self.n_target[1]

        L  = np.einsum( 'ij,ji', self.part.T, na + nb )
        L += np.einsum( 'ij,ji', self.part.V, na_target + nb_target - na - nb)
        L += np.einsum( 'ij,ji', self.guess, na_target - na)
        L += np.einsum( 'ij,ji', self.guess, nb_target - nb)
        L += np.einsum(   'i,i', v[:self.nauxbf], self.grad_a  )
        L += np.einsum(   'i,i', v[self.nauxbf:], self.grad_b  )
        
        #Add Regularization

        return L

    def invert(self, target_density):

        self.n_target = target_density
        
        if self.method.lower() == 'bfgs':

            opt_results = minimize( fun = self.lagrangian,
                                    x0  = self.v0, 
                                    jac = self.gradient,
                                    method = self.method,
                                    )

            # print("Im done optimizing")
            # print(opt_results.x)

            self.inverted_vks = opt_results.x
             



        