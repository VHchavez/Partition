"""
inverter.py
"""

import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt

from opt_einsum import contract

from pyscf import scf

from .util import to_grid, basis_to_grid

import psi4
# psi4.core.be_quiet()

import sys

class Inverter():

    def __init__(self, partition_object, 
                       opt_method='bfgs', ):

        self.part     = partition_object
        self.inv_type = "xc" if self.part.nfrags == 1 else "partition"
        self.opt_method = opt_method

        self.nauxbf   = self.part.nbf

        self.grad_a = None
        self.grad_b = None

        self.debug  = False

        self.c_target = None
        
        self.v = np.zeros( 2 * self.nauxbf )

        self.reg = 1e-16

        #Initialize
        if self.part.frags is None:
            self.part.scf(method="svwn")

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

        print("Energies")
        print(eigvecs)

        return C, Cocc, D, eigvecs
    
    # def build_orbitals(self, diag, ndocc):
    #     """Summary: Under a given b vector, construct a fock matrix and diagonalize it

    #     F = F0 + V(b)

    #     FC = SCE

    #     resulting mo_coeff(C), mo_energy(E), and density matrix are stored as instance attributes
    #     """

    #         # Diagonalize Fock matrix: [Szabo:1996] pp. 145              
    #     Fp = self.part.A.np.dot(diag).dot(self.part.A.np)                   # Eqn. 3.177
    #     e, C2 = np.linalg.eigh(Fp)             # Solving Eqn. 1.178
    #     C = self.part.A.np.dot(C2)                          # Back transform, Eqn. 3.174
    #     Cocc = C[:, :ndocc]                                                              
    #     D = np.einsum('pi,qi->pq', Cocc, Cocc) # [Szabo:1996] Eqn. 3.145, pp. 139


    #     return C, Cocc, D, e


    def initial_guess(self, guess):
        """
        Provides initial guess for inversion
        """ 

        print("Calculating Initial Guess for Potential Inversion")

        N = self.part.frags[0].natoms 

        if self.inv_type == "xc":
            if guess.lower() == "fermi_amaldi":

                if self.part.frags[0].Ca is None or self.part.frags[0].Cb is None:
                    raise ValueError("No Molecular Orbital Coefficient has been found, run a scf calculation")

                if self.c_target is not None:
                    print("CCSD target occupied")
                    Cocca = self.c_target[0]
                    Coccb = self.c_target[1]
                else:
                    Cocca = self.part.frags[0].Ca_occ
                    Coccb = self.part.frags[0].Cb_occ

                J, _ = self.part.form_jk( Cocca, Coccb )
                v_FA = - 1.0 * (1/N) * (J[0] + J[1])

                self.guess_a = - 1.0 * (1/(N/2)) * (J[0])
                self.guess_b = - 1.0 * (1/(N/2)) * (J[1])
 
            elif guess.lower() == "none":
                guess_matrix =   np.zeros_like(self.part.T) 

            elif guess.lower() == "dfa":

                psi4.set_options( { "reference" : "uhf" } )

                mol_guess = psi4.geometry(self.part.frags_str[0])
                _, wfn_guess = psi4.energy("pbe/"+self.part.basis_str, molecule=mol_guess, return_wfn=True)
                va_scf = wfn_guess.Vb().np.copy()
                vb_scf = wfn_guess.Vb().np.copy()

                self.vpot = wfn_guess.V_potential()

                J, _ = self.part.form_jk(self.c_target[0], self.c_target[1])

                na_target = psi4.core.Matrix.from_array( self.nt[0] )
                nb_target = psi4.core.Matrix.from_array( self.nt[1] )

                #Get VXC using exact density matrix
                wfn_guess.V_potential().set_D([ na_target, nb_target ])
                wfn_guess.V_potential().properties()[0].set_pointers( na_target, nb_target ) #In order to get it to the grid points
                va_target = psi4.core.Matrix.from_array( np.zeros_like( self.nt[0] ) )
                vb_target = psi4.core.Matrix.from_array( np.zeros_like( self.nt[1] ) )
                wfn_guess.V_potential().compute_V([va_target, vb_target])

                self.guess_a = J[0] + J[1] + va_target.np
                self.guess_b = J[0] + J[1] + vb_target.np

                if self.debug == True:
                    print("Hartree A\n")
                    print(J[0][:2,:2] + J[1][:2,:2])
                    print("\n")

                    print("Vxc with exact density\n")
                    print(va_target.np[:2,:2])
                    print("\n")

                    print("External Potential\n")
                    print("V\n",self.part.V[:2,:2])

                    print("Kinetic\n")
                    print(self.part.T[:2,:2])

                    print("Guess\n")
                    print(self.guess_a[:2,:2])

        else:
            raise ValueError("I am unable to generate the requested guess")

    def gradient(self, v):
        vks_a = np.einsum("ijk,k->ij", self.part.S3, v[:self.part.nbf]) + self.guess_a
        vks_b = np.einsum("ijk,k->ij", self.part.S3, v[self.part.nbf:]) + self.guess_b
        fock_a = self.part.V + self.part.T + vks_a 
        fock_b = self.part.V + self.part.T + vks_b

        Ca, Coca, Da, eigvecs_a = self.build_orbitals( fock_a, self.part.frags[0].nalpha )
        Cb, Cocb, Db, eigvecs_b = self.build_orbitals( fock_b, self.part.frags[0].nalpha )

        self.grad_a = np.einsum('ij,ijt->t', (Da-self.nt[0]), self.part.S3)
        self.grad_b = np.einsum('ij,ijt->t', (Db-self.nt[1]), self.part.S3) 

        self.grad = np.concatenate( (self.grad_a, self.grad_b) )

        return -self.grad

    def lagrangian(self, v):
        vks_a = np.einsum("ijk,k->ij", self.part.S3, v[:self.part.nbf]) + self.guess_a
        vks_b = np.einsum("ijk,k->ij", self.part.S3, v[self.part.nbf:]) + self.guess_b
        fock_a = self.part.V + self.part.T + vks_a 
        fock_b = self.part.V + self.part.T + vks_b



        Ca, Coca, Da, eigvecs_a = self.build_orbitals( fock_a, self.part.frags[0].nalpha )
        Cb, Cocb, Db, eigvecs_b = self.build_orbitals( fock_b, self.part.frags[0].nbeta )

        # sys.exit()

        if True:  #Plot potential at each step
            pb, v_guess = self.part.generate_1D_phi( self.nt[0] + self.nt[1], "pbe")

            vgrida = np.einsum('t,rt->r', v[:self.part.nbf], pb)
            vgridb = np.einsum('t,rt->r', v[self.part.nbf:], pb)


            plt.figure(figsize=(3,4))

            plt.plot(self.part.grid[:,0], vgrida + vgridb)
            # plt.plot(self.part.grid[:,0], vgridb)
            # plt.plot(self.part.grid[:,0], v_guess)

            plt.xlim(-0, 0.5)


            
            #USING PSI4 UNIQUELY
                        # vgrid = to_grid(v[:self.part.nbf], vpot=self.vpot)
            # vgrid, grid = basis_to_grid( v[:self.part.nbf], Vpot=self.vpot, blocks=False )
            # self.xyz_v = vgrid
            # self.xyz   = grid
            # plt.plot(vgrid)

            plt.show()


        self.grad_a = np.einsum('ij,ijt->t', (Da-self.nt[0]), self.part.S3)
        self.grad_b = np.einsum('ij,ijt->t', (Db-self.nt[1]), self.part.S3) 

        kinetic   =  np.einsum('ij,ji', self.part.T, (Da + Db))
        potential =  np.einsum('ij,ji', self.part.V, (Da + Db) - self.nt[0] - self.nt[1])
        potential += np.einsum('ij,ji', self.guess_a, (Da - self.nt[0])  )
        potential += np.einsum('ij,ji', self.guess_b, (Db - self.nt[1])  )
        optz      =  np.einsum('t,t', v[:self.nauxbf], self.grad_a)
        optz      += np.einsum('t,t', v[self.nauxbf:], self.grad_b)
    

        print(f" L1: {kinetic}, L2: {potential}, L3: {optz}")

        L1 = kinetic + potential + optz

        return -L1

    def invert(self, target_density, target_cocc=None, v_guess="dfa"):

        self.c_target = target_cocc
        self.nt = target_density
        self.initial_guess(v_guess)
    
        if self.opt_method.lower() == 'bfgs':

            opt_results = minimize( fun = self.lagrangian,
                                    x0  = self.v, 
                                    jac = self.gradient,
                                    method = self.opt_method,
                                    options = {"maxiter" : 100}
                                    )

            print("Was I successful???", opt_results.success)
            print("More info???", opt_results.message)

            self.v = opt_results.x
             



        