
from dask.distributed import Client
from multiprocessing import Pool, Array, Value
from dataclasses import dataclass
import numpy as np

import os, glob, sys

# psi4.core.be_quiet()

# import partition
from partition import wuyang

import psi4
psi4.core.be_quiet()


@dataclass
class pdft_fragment:
    pass




def fragment_scf(mol_string, potential=None, restricted="UKS"):

    import psi4
    # from partition import wuyang
    psi4.core.be_quiet()

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


if __name__ == '__main__':


    molecule = """
    He 0.0 0.0 0.0
    He 0.0 0.0 1.0
    symmetry c1
    """

    f1 = """
    He 0.0 0.0 0.0
    He 0.0 0.0 1.0
    symmetry c1
    """

    frags        = [molecule, f1]


    ############### Initial SCF Cycle for all fragments ########################


    c = Client()
    ret = c.map( fragment_scf, frags )
    frag_data = [ i.result() for i in ret ]

    print(frag_data[0].energy)
    print(frag_data[1].energy)
    

    ############### Process of Generting Partition Potential ########################

    # target_density = frag_data[0].Da + frag_data[0].Db
    # print("I found target density")
    # basis = wuyang(target_density, frag_data)

    # print(basis)


    #Generate a new external potential. 
    dummy_v1 = np.zeros_like( frag_data[0].Da )
    np.fill_diagonal(dummy_v1, dummy_v1.diagonal() + 0.1)

    dummy_v2 = np.zeros_like( frag_data[1].Da )
    np.fill_diagonal(dummy_v2, dummy_v2.diagonal() + 0.1)

    mols = [molecule, f1]
    vps  = [dummy_v1, dummy_v2]


    ############### Feed Partition Potential Back to Fragment ########################


    ret = c.map( fragment_scf, mols, vps )
    frag_data = [ i.result() for i in ret ]

    print(frag_data[0].energy)
    print(frag_data[1].energy)




    for f in glob.glob("psi.*.clean"):
        os.remove(f)


    c.close()