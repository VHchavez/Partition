

from multiprocessing import Pool, Array, Value
from dataclasses import dataclass
import numpy as np

import os, glob, sys

# psi4.core.be_quiet()

# import partition
# from partition import wuyang


@dataclass
class pdft_fragment:
    pass


def wuyang(target_density, frag_info):

    if len(frag_info) == 1:
        mol_string = frag_info[0]
        mol = psi4.geometry(mol_string)

    #Basis set
    basis   = psi4.core.BasisSet.build( mol_string, key="BASIS", target='cc-pvdz')

    return basis

def fragment_scf_class(mol_string, potential=None, restricted="UKS"):

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


    energy = wfn.get_energies("Total Energy")
    print("Energy of fragment", energy)

    return frag_info
def fragment_scf_unrestricted(mol_string, potential=None):

    import psi4
    psi4.core.be_quiet()

    mol = psi4.geometry(mol_string)
    wfn_base = psi4.core.Wavefunction.build(mol, "cc-pvdz")
    wfn = psi4.proc.scf_wavefunction_factory("svwn", wfn_base, "UKS")
    wfn.initialize()

    if potential is not None:
        print("I am adding external potential")
        wfn.H().np[:] -= potential


    wfn.iterations()


    # what happens if I skip finalize energy?
    # wfn.finalize_energy()

    energy = wfn.get_energies("Total Energy")

    print(f"Energy of fragment: {energy}")


    return [wfn.Da().np, wfn.Db().np, wfn.Ca().np, wfn.Cb().np, wfn.epsilon_a().np, wfn.epsilon_b().np]
def fragment_scf_restricted(mol_string, potential=None):

    import psi4
    psi4.core.be_quiet()

    mol = psi4.geometry(mol_string)
    wfn_base = psi4.core.Wavefunction.build(mol, "cc-pvdz")
    wfn = psi4.proc.scf_wavefunction_factory("svwn", wfn_base, "RKS")

    if potential is not None:
        wfn.H().np[:] -= potential

    wfn.initialize()
    wfn.iterations()

    return [wfn.Da().np, wfn.Ca().np, wfn.epsilon_a().np]


if __name__ == '__main__':


    if True:
        # f1 = """
        # @He 0.0 0.0 0.0
        # He 0.0 0.0 1.0
        # symmetry c1
        # """

        # f2 = """
        # He 0.0 0.0 0.0
        # @He 0.0 0.0 1.0
        # symmetry c1
        # """
        pass

    molecule = """
    He 0.0 0.0 0.0
    He 0.0 0.0 1.0
    symmetry c1
    """

    frags        = [molecule]


    ############### Initial SCF Cycle for all fragments ########################

    p = Pool(processes=len(frags))
    frag_data = p.map( fragment_scf_class, frags)
    assert len(frags) == len(frag_data), "Number of fragments do not match number of results"
    p.close()
    p.join()


    ############### Process of Generting Partition Potential ########################

    target_density = frag_data[0].Da + frag_data[0].Db
    print("I found target density")
    basis = wuyang(target_density, frags[0])

    # print(basis)

    print("Hello")

    sys.exit()


    #Generate a new external potential. 
    dummy_v1 = np.zeros_like( frag_data[0].Da )
    np.fill_diagonal(dummy_v1, dummy_v1.diagonal() + 0.1)

    dummy_v2 = np.zeros_like( frag_data[1].Da )
    np.fill_diagonal(dummy_v2, dummy_v2.diagonal() + 0.1)

    vp_map = [(He1, dummy_v1), (He2, dummy_v2)]


    ############### Feed Partition Potential Back to Fragment ########################


    p = Pool(processes=len(frags))
    frag_results = p.starmap( fragment_scf_class, vp_map)
    assert len(frags) == len(frag_results), "Number of fragments do not match number of results"
    p.close()
    p.join()


    for f in glob.glob("psi.*.clean"):
        os.remove(f)