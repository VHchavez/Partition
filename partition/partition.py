"""
partition.py
"""

import psi4
from dask.distributed import Client

from dataclasses import dataclass
@dataclass
class pdft_fragment:
    pass

def _scf(mol_string, 
         potential=None, 
         restricted="UKS"):

    frag_info = pdft_fragment()

    mol = psi4.geometry(mol_string)
    wfn_base = psi4.core.Wavefunction.build(mol, "cc-pvdz")
    wfn = psi4.proc.scf_wavefunction_factory("svwn", wfn_base, restricted)
    wfn.initialize()

    if potential is not None:
        wfn.H().np[:] += potential


    wfn.iterations()
    wfn.finalize_energy()

    #Paste results to pdf_fragment
    energies = { "enuc" : wfn.get_energies('Nuclear'),
                "e1"   : wfn.get_energies('One-Electron'),
                "e2"   : wfn.get_energies('Two-Electron'),
                "exc"  : wfn.get_energies('XC'),
                "total": wfn.get_energies('Total Energy')}

    frag_info.mol_str  = mol_string
    frag_info.Da       = wfn.Da().np
    frag_info.Db       = wfn.Db().np
    frag_info.Ca       = wfn.Ca().np
    frag_info.Cb       = wfn.Cb().np
    frag_info.eig_a    = wfn.epsilon_a().np
    frag_info.eig_b    = wfn.epsilon_b().np
    frag_info.energies = energies
    frag_info.energy   = wfn.get_energies('Total Energy')

    return frag_info


class Partition():
    def __init__(self, frag_info, method, basis):
        # self.f_info = frag_info
        self.method     = method
        self.basis_str  = basis
        self.frags      = frag_info
        self.client     = Client()

        self.basis      = None

        #Initialized Methods
        self.build_basis()
        #self.basis = None

    def scf(self,
            vext   = None,):

        frags = self.frags

        #Check wether or not there is a vext to be added for scf cycle
        if vext is None:
            ret = self.client.map( _scf, frags )
        else:
            ret = self.client.map( _scf, frags, vext )

        frag_data = [ i.result() for i in ret ]

        return frag_data

    def build_basis(self):
        """
        Creates basis information for all calculations
        """
        mol   = psi4.geometry( self.frags[0] )
        basis = psi4.core.BasisSet.build( mol, key='BASIS', target="cc-pvdz" )

        self.basis = basis

        return