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

    if potential is not None:
        potential = [psi4.core.Matrix.from_array(i) for i in potential]
        wfn.iterations(pdft=True, pdft_matrix=potential)
    else:
        wfn.iterations()
    wfn.finalize_energy()


    # if potential is not None:
    #     exc = generate_exc( mol_string, basis, wfn.Da().np )

    #Paste results to pdf_fragment
    energies = {"enuc" : wfn.get_energies('Nuclear'),
                "e1"   : wfn.get_energies('One-Electron'),
                "e2"   : wfn.get_energies('Two-Electron'),
                "exc"  : wfn.get_energies('XC'),
                "total": wfn.get_energies('Total Energy')
                }

    print("Initial Energy:", energies["total"])

    # if potential is not None:
    #     pass
    #     # energies["exc"] = exc

    #     # full_matrix = wfn.Da().np + wfn.Db().np 
    #     # full_matrix = psi4.core.Matrix.from_array( full_matrix )

    #     # p = psi4.core.Matrix.from_array( (potential[0] + potential[1])/2.0 )

    #     # # print("How much am I removing from Core matrix", (p0.vector_dot(full_matrix) +  p1.vector_dot( full_matrix ) ))

    #     # energies["e1"] -= (p.vector_dot(full_matrix) )
 
    frag_info.geometry = mol.geometry().np
    frag_info.natoms   = mol.natom()
    frag_info.nalpha   = wfn.nalpha()
    frag_info.nbeta    = wfn.nbeta()
    frag_info.mol_str  = mol_string
    frag_info.Da       = wfn.Da().np
    frag_info.Db       = wfn.Db().np
    frag_info.Ca       = wfn.Ca().np
    frag_info.Cb       = wfn.Cb().np
    frag_info.Va       = wfn.Va().np
    frag_info.Vb       = wfn.Vb().np
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
    def __init__(self, basis, mol_str, frags_str=None):
    
        self.basis_str  = basis
        self.mol_str    = mol_str
        self.mol        = None
        self.frags_str  = frags_str
        self.frags      = None
        self.nfrags     = 0 if frags_str is None else len( frags_str )
        if self.nfrags == 1:
            raise ValueError("Number of fragments cannot be equal to one!")


        #Client for Paralellization on fragments
        self.client     = Client()

        #Allocate Core matrices for Fragmetns
        self.Ts         = []
        self.Vs         = []

        #Generate Basis Set
        self.build_basis()
        self.nbf        = self.basis.nbf()
        self.generate_mints_matrices()
        self.generate_core_matrices()

        #Plotting Grid
        self.generate_grid()

        #Initial Guess scf
        self.scf_mol()
        if self.nfrags != 0:
            self.scf_frags()

        #Generate invert class
        self.inverter = Inverter(self)

    ############ METHODS ############

    def build_basis(self):
        """
        Creates basis information for all calculations
        """
        print("Creating Molecule basis")
        mol = psi4.geometry(self.mol_str)
        basis = psi4.core.BasisSet.build( mol, key='BASIS', target=self.basis_str)
        self.basis = basis

        if self.nfrags > 1:
            frags_basis = []
            for i in range(self.nfrags):
                frag   = psi4.geometry( self.frags_str[i] )
                basis = psi4.core.BasisSet.build( frag, key='BASIS', target=self.basis_str)
                frags_basis.append(basis)

            self.frags_basis = frags_basis

    def generate_core_matrices(self):
        mints_mol = psi4.core.MintsHelper( self.basis )
        self.T = mints_mol.ao_kinetic().np.copy()
        self.V = mints_mol.ao_potential().np.copy()

        if self.nfrags > 1:
            for i in range(self.nfrags):
                frag_mol = psi4.core.MintsHelper( self.frags_basis[i] )
                self.Ts.append( frag_mol.ao_kinetic().np.copy()   )
                self.Vs.append( frag_mol.ao_potential().np.copy() ) 

    def generate_mints_matrices(self):
        mints = psi4.core.MintsHelper( self.basis )

        self.S = mints.ao_overlap().np
        A = mints.ao_overlap()
        A.power(-0.5, 1.e-14)
        self.A = A
        self.S3 = np.squeeze(mints.ao_3coverlap(self.basis,self.basis,self.basis))
        self.jk = None

    def generate_grid(self):
        coords = []
        for x in np.linspace(-7, 7, 10001):
            coords.append((x, 0., 0.))
        coords = np.array(coords)

        self.grid = coords

    def generate_1D_phi(self, atom, target, functional):

        mol = gto.M(atom=atom,
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
        
    def scf_mol(self, 
               method="svwn"):
        
        psi4.set_options({"maxiter" : 100})
        ret = self.client.map( _scf, [self.mol_str], [method], [self.basis_str] )
        data = [i.result() for i in ret]
        self.mol = data[0]

    def scf_frags(self,
            method = "svwn",
            vext   = None,
            evaluate = False):

        frags     = self.frags_str 
        method_it = [method         for i in range(self.nfrags)]
        basis_it  = [self.basis_str for i in range(self.nfrags)]
        vext_it   = [vext           for i in range(self.nfrags)]

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


    ##### PDFT METHODS #####

    def frag_sum(self):
        n_suma = np.zeros( (self.nbf, self.nbf) )
        n_sumb = np.zeros( (self.nbf, self.nbf) )
        coc_suma = np.zeros( (self.nbf, self.nbf) )
        coc_sumb = np.zeros( (self.nbf, self.nbf) )

        # assert self.nfrags == 2, "Nfrags is different to number of fragments"

        for i in range(self.nfrags):
            n_suma += self.frags[i].Da
            n_sumb += self.frags[i].Db
            coc_suma += self.frags[i].Ca_occ
            coc_sumb += self.frags[i].Cb_occ

        self.frags_na = n_suma.copy()
        self.frags_nb = n_sumb.copy()
        self.frags_coca = coc_suma
        self.frags_cocb = coc_sumb
