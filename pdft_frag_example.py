
from dask.distributed import Client
from multiprocessing import Pool, Array, Value

import numpy as np

import os, glob, sys

from partition import Partition

import psi4
psi4.core.be_quiet()






# def _scf(mol_string, 
#                 potential=None, restricted="UKS"):

#     frag_info = pdft_fragment()

#     mol = psi4.geometry(mol_string)
#     wfn_base = psi4.core.Wavefunction.build(mol, "cc-pvdz")
#     wfn = psi4.proc.scf_wavefunction_factory("svwn", wfn_base, restricted)
#     wfn.initialize()

#     if potential is not None:
#         wfn.H().np[:] += potential


#     wfn.iterations()
#     wfn.finalize_energy()

#     #Paste results to pdf_fragment
#     energies = { "enuc" : wfn.get_energies('Nuclear'),
#                  "e1"   : wfn.get_energies('One-Electron'),
#                  "e2"   : wfn.get_energies('Two-Electron'),
#                  "exc"  : wfn.get_energies('XC'),
#                  "total": wfn.get_energies('Total Energy')}

#     frag_info.mol_str  = mol_string
#     frag_info.Da       = wfn.Da().np
#     frag_info.Db       = wfn.Db().np
#     frag_info.Ca       = wfn.Ca().np
#     frag_info.Cb       = wfn.Cb().np
#     frag_info.eig_a    = wfn.epsilon_a().np
#     frag_info.eig_b    = wfn.epsilon_b().np
#     frag_info.energies = energies
#     frag_info.energy   = wfn.get_energies('Total Energy')

#     return frag_info

# def scf(frags,  vext = None,
#                 client = None):

#     #Determine wheter or not a client has been initializied
#     if client is None:
#         client = Client()

#     #Check wether or not there is a vext to be added for scf cycle
#     if vext is None:
#         ret = client.map( _scf, frags )
#     else:
#         ret = client.map( _scf, frags, vext )

#     frag_data = [ i.result() for i in ret ]

#     return frag_data


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

    frags        = [molecule]

    part = Partition( frags, "svwn", "cc-pvdz" )
    frag_data = part.scf()



    ############### Process of Generting Partition Potential ########################

    #Generate a new external potential. 
    dummy_v1 = np.zeros_like( frag_data[0].Da )
    np.fill_diagonal(dummy_v1, dummy_v1.diagonal() + 0.1)

    mols = [molecule]
    vps  = [dummy_v1]


    ############### Feed Partition Potential Back to Fragment ########################

    frag_data = part.scf( vps )

    for f in glob.glob("psi.*.clean"):
        os.remove(f)

    # for f in glob.glob("dask-worker-space"):
    #     os.rmdir(f) 

    print("I'm leaving victorious")

    part.client.close()