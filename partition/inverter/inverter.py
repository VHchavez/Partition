"""
inverter.py
"""

import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt

from opt_einsum import contract

# from pyscf import scf

from .methods.wuyang import WuYang
from .methods.mrks import MRKS

# from .util import to_grid, basis_to_grid, get_from_grid

import psi4
psi4.core.be_quiet()

import sys

class Inverter(WuYang, MRKS):

    def __init__(self, partition_object,
                       debug=False):

        self.part     = partition_object
        self.inv_type = "partition"
        self.opt_method = None

        # self.nauxbf   = self.part.nbf

        self.grad_a = None
        self.grad_b = None

        #Target Quantities
        self.ct = None
        self.nt = None

        #Inverted Potential
        self.v = np.zeros( 2 * self.nauxbf )
        self.reg = 0.0

        self.debug  = debug

    #PDFT INVERSION ###################################################################################

    def invert(self, method, nt, ct=None, initial_guess="none", 
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
        self.ct = ct
        self.nt = nt
        self.initial_guess(initial_guess)
        self.opt_method = opt_method

        if method.lower() == "wuyang":
            self.wuyang_invert(opt_method, initial_guess)
        elif method.lower() == 'mrks':
            self.mrks_invert(opt_max_iter)
            pass


    def initial_guess(self, guess):
        """
        Provides initial guess for inversion
        """ 

        cocca_0 = psi4.core.Matrix.from_array(self.ct[0]) 
        coccb_0 = psi4.core.Matrix.from_array(self.ct[1]) 
        self.J0, _ = self.part.form_jk(cocca_0, coccb_0)

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
