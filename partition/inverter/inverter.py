"""
inverter.py
"""

from typing import List
from dataclasses import dataclass
import numpy as np
from pydantic import validator, BaseModel
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from opt_einsum import contract

# from pyscf import scf

from .methods.wuyang import WuYang
# from .methods.mrks import MRKS
from .methods.oucarter import Oucarter

# from .util import to_grid, basis_to_grid, get_from_grid

import psi4
import sys

@dataclass
class Plotter:    
    pass


class InverterOptions(BaseModel):
    verbose : bool = True

class Inverter(WuYang, Oucarter):

    def __init__(self, mol, basis, ref, frags, grid=None, jk=None, optInv={}):

        # Validate options
        optInv = {k.lower(): v for k, v in optInv.items()}
        for i in optInv.keys():
            if i not in InverterOptions().dict().keys():
                raise ValueError(f"{i} is not a valid option for Inverter")
        optInv = InverterOptions(**optInv)
        self.optInv = optInv

        self.mol       = mol
        self.basis     = basis
        self.basis_str = self.basis.name()
        mints = psi4.core.MintsHelper( self.basis )
        self.S3 = np.squeeze(mints.ao_3coverlap(self.basis,self.basis,self.basis))
        A = mints.ao_overlap()
        A.power( -0.5, 1e-16 )
        self.A = np.array(A)
        self.T = np.array(mints.ao_kinetic())
        self.V = np.array(mints.ao_potential())

        self.nbf      = self.basis.nbf()
        self.nauxbf   = self.basis.nbf()
        self.ref      = ref
        self.frags    = frags
        self.nfrags   = len(frags)

        self.nalpha = None
        self.nbeta  = None
        
        # Opt quantities
        self.opt_method = None
        self.grad_a = None
        self.grad_b = None

        #Target Quantities
        self.ct = None
        self.dt = None

        #Inverted Potential
        self.v0 = np.zeros( 2 * self.nauxbf )
        
        self.Plotter = Plotter()
        self.grid = grid
        self.jk   = jk
        # if self.ref == 1:
        #     self.v = np.zeros( 1 * self.nauxbf )
        # else:
        #     self.v = np.zeros( 1 * self.nauxbf )

        self.reg = 0.0


    #PDFT INVERSION ###################################################################################

    def invert(self, method, initial_guess="none", 
                            opt_max_iter=100,
                            opt_method='trust-krylov'):
        """
        Inversion procedure routing

        ct: np.array
            Target occupied orbitals. shape (nbf, nbf)
        nt: np.array
            Target density. shape (nbf, nbf)
        intitial guess: str, opt
            Initial guess for optimizer. Default: "none"
        opt_method: str, opt
            Method for WuYang Inversion through scipy optimizer. 
            Default: "trust-krylov"
        """

        self.cumulative_vs = []

        self.initial_guess(initial_guess)
        self.opt_method = opt_method

        if method.lower() == "wuyang":
            self.wuyang_invert(opt_method=opt_method, initial_guess='none')

        if method.lower() == 'oucarter':

            vxc_dft, vxc_plotter = self.oucarter(5, self.grid.plot_points)
            return vxc_dft, vxc_plotter

    def initial_guess(self, guess):
        """
        Provides initial guess for inversion
        """ 

        if guess.lower() == 'none':
            self.va = np.zeros_like(self.dt[0])
            self.vb = np.zeros_like(self.dt[0])

        else:
            cocca_0 = psi4.core.Matrix.from_array(self.ct[0]) 
            coccb_0 = psi4.core.Matrix.from_array(self.ct[1]) 
            self.J0, _ = self.part.form_jk(cocca_0, coccb_0)

        # if self.inv_type == "xc":

        #     print("Generating Initial Guess")

        # elif self.inv_type == 'partition':

        #     #Let us try to do same guess. This guess will first be FA for the whole system. 
        #     #If this doesn't work I'll try sum of fermi amaldis. 
        #     N = self.part.mol.nalpha + self.part.mol.nbeta

        #     if guess.lower() == 'fa':
        #         print("Calculating Fermi Amaldi")
        #         for f_i in self.part.frags:
        #             N = f_i.nalpha + f_i.nbeta
        #             coca = psi4.core.Matrix.from_array(f_i.Ca_occ)
        #             cocb = psi4.core.Matrix.from_array(f_i.Cb_occ)
        #             J, _ = self.part.form_jk(  coca , cocb )
        #             fa =  (- 1.0 / N) * (J[0] + J[1])
        #             self.va = fa
        #             self.vb = fa

        #     elif guess.lower() == 'hfa':
        #         print("Calculating Hartree Fermi Amaldi")
        #         for f_i in self.part.frags:
        #             N = f_i.nalpha + f_i.nbeta
        #             coca = psi4.core.Matrix.from_array(f_i.Ca_occ)
        #             cocb = psi4.core.Matrix.from_array(f_i.Cb_occ)
        #             J, _ = self.part.form_jk(  coca , cocb )
        #             fa =  (1 - 1.0 / N) * (J[0] + J[1])
        #             self.va = fa
        #             self.vb = fa

    
    def generate_jk(self, K=True, memory=2.50e8):
        jk = psi4.core.JK.build(self.basis)
        jk.set_memory(int(memory)) #1GB
        jk.set_do_K(K)
        jk.initialize()

        self.jk = jk

    def form_jk(self, C_occ_a, C_occ_b):
        if self.jk is None:
            self.generate_jk()

        C_occ_a = psi4.core.Matrix.from_array(C_occ_a)
        C_occ_b = psi4.core.Matrix.from_array(C_occ_b)

        self.jk.C_left_add(C_occ_a)
        self.jk.C_left_add(C_occ_b)
        self.jk.compute()
        self.jk.C_clear()

        Ja = self.jk.J()[0].np
        Jb = self.jk.J()[1].np
        J = [Ja, Jb]

        Ka = self.jk.K()[0].np
        Kb = self.jk.K()[1].np
        K = [Ka, Kb]

        return J, K