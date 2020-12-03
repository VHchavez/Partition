"""
partition.py
"""

import numpy as np

import psi4
from dask.distributed import Client

from .inverter import Inverter



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

    if potential is not None:
        wfn.H().np[:] += potential[0] + potential[1]


    wfn.iterations()
    wfn.finalize_energy()

    #Paste results to pdf_fragment
    energies = { "enuc" : wfn.get_energies('Nuclear'),
                "e1"   : wfn.get_energies('One-Electron'),
                "e2"   : wfn.get_energies('Two-Electron'),
                "exc"  : wfn.get_energies('XC'),
                "total": wfn.get_energies('Total Energy')}

    frag_info.geometry = mol.geometry().np
    frag_info.natoms   = mol.natom()
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

    # if potential is None:
    #     print("energies", energies)

    # print("fragment energy", frag_info.energy)

    return frag_info


class Partition():
    def __init__(self, frags_str, basis):
    
        self.basis_str  = basis
        self.frags_str  = frags_str
        self.frags      = None
        self.nfrags     = len( frags_str )
        self.client     = Client()


        #Generate Basis Set
        self.basis      = self.build_basis()
        self.nbf        = self.basis.nbf()
        self.generate_mints_matrices()


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
        self.T = mints.ao_kinetic().np
        self.V = mints.ao_potential().np
        self.S3 = np.squeeze(mints.ao_3coverlap(self.basis,self.basis,self.basis))
        print("Warning: Assume auxiliary basis set is equal to ao basis set")
        self.jk = None

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

        if evaluate != False:
            psi4.set_options({"maxiter" : 100})
        else:
            psi4.set_options({"maxiter" : 1})

        #Check wether or not there is a vext to be added for scf cycle
        if vext is None:
            ret = self.client.map( _scf, frags, method_it, basis_it )
        else:
            ret = self.client.map( _scf, frags, method_it, basis_it, vext_it )

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