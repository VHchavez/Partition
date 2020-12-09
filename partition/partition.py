"""
partition.py
"""

import numpy as np

import psi4
from dask.distributed import Client

from .inverter import Inverter
from .util import generate_exc

from pyscf import dft, gto
from kspies import util



from dataclasses import dataclass
@dataclass
class bucket:
    """
    Basic data class
    """
    pass

def _scf(mol_string, 
         method='svwn',
         basis='cc-pvdz',
         potential=None, 
         restricted="UKS"):

    frag_info = bucket()

    mol = psi4.geometry(mol_string)
    wfn_base = psi4.core.Wavefunction.build(mol, basis)
    wfn = psi4.proc.scf_wavefunction_factory(method, wfn_base, restricted)
    wfn.initialize()

    print(help(wfn.iterations))

    # if potential is not None:
    #     wfn.H().np[:] += (potential[0] + potential[1])/2.0

    if potential is not None:
        potential = [psi4.core.Matrix.from_array(i) for i in potential]
        wfn.iterations(pdft=True, pdft_matrix=potential[0])
    else:
        wfn.iterations()
    wfn.finalize_energy()


    # if potential is not None:
    #     exc = generate_exc( mol_string, basis, wfn.Da().np )

    #Paste results to pdf_fragment
    energies = { "enuc" : wfn.get_energies('Nuclear'),
                "e1"   : wfn.get_energies('One-Electron'),
                "e2"   : wfn.get_energies('Two-Electron'),
                "exc"  : wfn.get_energies('XC'),
                "total": wfn.get_energies('Total Energy')
                }

    print("Initial Energy:", energies["total"])

    if potential is not None:
        pass
        # energies["exc"] = exc

        # full_matrix = wfn.Da().np + wfn.Db().np 
        # full_matrix = psi4.core.Matrix.from_array( full_matrix )

        # p = psi4.core.Matrix.from_array( (potential[0] + potential[1])/2.0 )

        # # print("How much am I removing from Core matrix", (p0.vector_dot(full_matrix) +  p1.vector_dot( full_matrix ) ))

        # energies["e1"] -= (p.vector_dot(full_matrix) )
 
    frag_info.geometry = mol.geometry().np
    frag_info.natoms   = mol.natom()
    frag_info.nalpha   = wfn.nalpha()
    frag_info.nbeta    = wfn.nbeta()
    frag_info.mol_str  = mol_string
    frag_info.Da       = wfn.Da().np
    frag_info.Db       = wfn.Db().np
    frag_info.Ca       = wfn.Ca().np
    frag_info.Cb       = wfn.Cb().np
    frag_info.Ca_occ   = wfn.Ca_subset("AO", "OCC").np
    frag_info.Cb_occ   = wfn.Cb_subset("AO", "OCC").np
    frag_info.Ca_vir   = wfn.Ca_subset("AO", "VIR").np
    frag_info.Cb_vir   = wfn.Cb_subset("AO", "VIR").np
    frag_info.eig_a    = wfn.epsilon_a().np
    frag_info.eig_b    = wfn.epsilon_b().np
    frag_info.energies = energies
    frag_info.energy   = wfn.get_energies('Total Energy')

    return frag_info


class Partition():
    def __init__(self, frags_str, basis):
    
        self.basis_str  = basis
        self.frags_str  = frags_str
        self.frags      = None
        self.nfrags     = len( frags_str )
        self.client     = Client()


        #Generate Basis Set
        self.basis      = self.build_basis(self.basis_str)
        self.nbf        = self.basis.nbf()
        self.generate_mints_matrices()


        #Plotting Grid
        self.generate_grid()

        #Initial Guess scf
        self.scf()

        #Generate invert class
        self.inverter = Inverter(self)

        # self.generate_jk()
        #Generate Matrices


        #Initialized Methods
        #self.basis = None
        #self.mints = None

    def generate_mints_matrices(self):
        mints = psi4.core.MintsHelper( self.basis )

        self.S = mints.ao_overlap().np
        A = mints.ao_overlap()
        A.power(-0.5, 1.e-14)
        self.A = A
        self.T = mints.ao_kinetic().np.copy()
        self.V = mints.ao_potential().np.copy()
        self.S3 = np.squeeze(mints.ao_3coverlap(self.basis,self.basis,self.basis))
        print("Warning: Assume auxiliary basis set is equal to ao basis set")
        self.jk = None

    def generate_grid(self):
        coords = []
        for x in np.linspace(0, 10, 1001):
            coords.append((x, 0., 0.))
        coords = np.array(coords)

        self.grid = coords

    def generate_1D_phi(self, target, functional):

        mol = gto.M(atom="He",
                    basis=self.basis_str)
                    
        pb = dft.numint.eval_ao( mol, self.grid )

        guess_grid = util.eval_vxc(mol, target, functional, self.grid)

        return pb, guess_grid

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
        
    def scf(self,
            method = "svwn",
            vext   = None,
            evaluate = False):

        frags = self.frags_str
        method_it = [method for i in range(self.nfrags)]
        basis_it  = [self.basis_str for i in range(self.nfrags)]
        vext_it   = [vext for i in range(self.nfrags)]

        assert len( method_it ) == len( basis_it ) == len( frags )

        if evaluate == False:
            psi4.set_options({"maxiter" : 100})
        else:
            print("Just evaluating")
            psi4.set_options({"maxiter" : 1})

        #Check wether or not there is a vext to be added for scf cycle
        if vext is None:
            ret = self.client.map( _scf, frags, method_it, basis_it )
        else:
            ret = self.client.map( _scf, frags, method_it, basis_it, vext_it, )

        frag_data = [ i.result() for i in ret ]

        self.frags = frag_data

    def build_auxbasis(self, aux_basis_str):
        self.aux_basis = self.build_basis( aux_basis_str )

    def build_basis(self, basis=None):
        """
        Creates basis information for all calculations
        """
        mol   = psi4.geometry( self.frags_str[0] )

        if basis is None:
            basis = psi4.core.BasisSet.build( mol, key='BASIS', target="cc-pvdz" )
        else: 
            basis = psi4.core.BasisSet.build( mol, key='BASIS', target=basis     )
        return basis