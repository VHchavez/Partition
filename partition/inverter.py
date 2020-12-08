"""
inverter.py
"""

import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt

import psi4
psi4.core.be_quiet()

class Inverter():

    def __init__(self, partition_object, 
                       opt_method='bfgs', ):

        self.part     = partition_object
        self.inv_type = "xc" if self.part.nfrags == 1 else "partition"
        self.opt_method = opt_method

        self.nauxbf   = self.part.nbf

        self.grad_a = None
        self.grad_b = None

        self.c_target = None
        
        self.v0 = np.zeros( 2 * self.nauxbf )

        self.reg = 1e-16

        #Initialize
        if self.part.frags is None:
            self.part.scf(method="hf")


    def build_orbitals(self, diag, ndocc):

        diag = psi4.core.Matrix.from_array( diag )
        Fp = psi4.core.triplet(self.part.A, diag, self.part.A, True, False, True)
        nbf = self.part.A.shape[0]
        Cp = psi4.core.Matrix(nbf, nbf)
        eigvecs = psi4.core.Vector(nbf)
        Fp.diagonalize(Cp, eigvecs, psi4.core.DiagonalizeOrder.Ascending)
        C = psi4.core.doublet(self.part.A, Cp, False, False)
        Cocc = psi4.core.Matrix(nbf, ndocc)
        Cocc.np[:] = C.np[:, :ndocc]
        D = psi4.core.doublet(Cocc, Cocc, False, True)

        C       = C.np
        Cocc    = Cocc.np
        D       = D.np
        eigvecs = eigvecs.np

        grad = (self.nt[0] + self.nt[1]) - D
        grad = np.einsum(  "ij,ijk->k", grad, self.part.S3  )
        self.grad = np.concatenate(  (grad, grad) )

        return C, Cocc, D, eigvecs
    
    def initial_guess(self, guess):
        """
        Provides initial guess for inversion
        """ 

        N = self.part.frags[0].natoms 

        if self.inv_type == "xc":

            if guess.lower() == "fermi_amaldi":

                if self.part.frags[0].Ca is None or self.part.frags[0].Cb is None:
                    raise ValueError("No Molecular Orbital Coefficient has been found, run a scf calculation")

                print("This is my c_target", self.c_target)

                if self.c_target is not None:
                    print("CCSD target occupied")
                    Cocca = self.c_target[0]
                    Coccb = self.c_target[1]
                else:
                    Cocca = self.part.frags[0].Ca_occ
                    Coccb = self.part.frags[0].Cb_occ

                J, _ = self.part.form_jk( Cocca, Coccb )
                v_FA = - 1.0 * (1/N) * (J[0] + J[1])

                guess_matrix = v_FA
            
            elif guess.lower() == "core":
                guess_matrix =   self.part.T + self.part.V

            elif guess.lower() == "dfa":
                mol_guess = psi4.geometry(self.part.frags_str[0])
                _, wfn_guess = psi4.energy("pbe/"+self.part.basis_str, molecule=mol_guess, return_wfn=True)
                vxc = wfn_guess.Va().np + wfn_guess.Vb().np

                J, _ = self.part.form_jk(self.c_target[0], self.c_target[1])

                # Cocca = self.c_target[0]
                # Coccb = self.c_target[1]
                # J, _ = self.part.form_jk( Cocca, Coccb )
                # v_fa = - 1.0 * (1/N) * (J[0] + J[1])

            #     guess_matrix = vxc + J[0] + J[1] #+ v_fa * 0.0
            # self.guess = guess_matrix

                self.guess_a = wfn_guess.Va().np + J[0]
                self.guess_b = wfn_guess.Vb().np + J[1]

        else:
            raise ValueError("I am unable to generate the requested guess")

    def gradient(self, v):
        #Generate Kohn Sham Potential
        print("Running Gradient")
        vxc_a = np.einsum("ijk,k->ij", self.part.S3, v[:self.part.nbf]) + self.guess_a
        vxc_b = np.einsum("ijk,k->ij", self.part.S3, v[self.part.nbf:]) + self.guess_b
        fock = self.part.V + self.part.T + (vxc_a + vxc_b) 

        C, Cocc, D, eigvecs = self.build_orbitals( fock, 10 )

        return - self.grad

    def lagrangian(self, v):
        print("Running Lagrangian")
        vxc_a = np.einsum("ijk,k->ij", self.part.S3, v[:self.part.nbf]) + self.guess_a
        vxc_b = np.einsum("ijk,k->ij", self.part.S3, v[self.part.nbf:]) + self.guess_b
        fock = self.part.V + self.part.T + (vxc_a + vxc_b) 

        C, Cocc, D, eigvecs = self.build_orbitals( fock, 10 )
        print("Energies", eigvecs[:10])

        if True:  #Plot potential at each step
            pb = self.part.generate_1D_phi()

            vgrida = np.einsum('t,rt->r', v[:self.part.nbf], pb)
            vgridb = np.einsum('t,rt->r', v[self.part.nbf:], pb)

            plt.plot(self.part.grid[:,0], vgrida+vgridb)
            plt.show()

        kinetic   = np.einsum( 'ij,ji', self.part.T, D )
        potential = np.einsum( 'ij,ji', self.part.V + self.guess_a + self.guess_b, (self.nt[0] - self.nt[1]) - D)
        optz      = np.einsum(   'i,i', v, self.grad  )

        print(f" L1: {kinetic}, L2: {potential}, L3: {optz}")

        L1 = kinetic + potential + optz

        return -L1

    def invert(self, target_density, target_cocc=None, v_guess="dfa"):

        self.c_target = target_cocc
        self.nt = target_density
        self.initial_guess(v_guess)
    
        if self.opt_method.lower() == 'bfgs':

            opt_results = minimize( fun = self.lagrangian,
                                    x0  = self.v0, 
                                    jac = self.gradient,
                                    method = self.opt_method,
                                    options = {"maxiter" : 1}
                                    )

                # print("Im done optimizing")
                # print(opt_results.x)

            print(opt_results.__dict__)

            self.inverted_vks = opt_results.x
             



        