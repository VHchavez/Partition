"""
fragments.py
Handles individual fragment informations. 
Behaves like a KohnSham object in pyCADMium
"""
from types import MethodDescriptorType
from copy import copy

import numpy as np
import psi4
from dataclasses import dataclass
from pydantic import validator, BaseModel
from opt_einsum import contract

from ..grid import Grider

@dataclass
class V: # Implicityly on the basis set
    vnuc : np.ndarray = np.empty((0,0))
    vext : np.ndarray = np.empty((0,0))
    vhxc : np.ndarray = np.empty((0,0))
    veff : np.ndarray = np.empty((0,0))
    vkin : np.ndarray = np.empty((0,0))
    vh   : np.ndarray = np.empty((0,0))
    vx   : np.ndarray = np.empty((0,0))
    vc   : np.ndarray = np.empty((0,0))

@dataclass
class Plotter:
    pass

class Vnm_fragment:
    pass

@dataclass
class E:
    kin  : float = 0.0
    ext  : float = 0.0
    har  : float = 0.0
    xc   : float = 0.0
    nuc  : float = 0.0
    tot  : float = 0.0
    E0   : float = 0.0
    # E   : float = 0.0
    # Ec  : float = 0.0
    # Ex  : float = 0.0
    # Exc : float = 0.0
    # Eks : np.ndarray = np.empty((0,0)) 
    # Vks : np.ndarray = np.empty((0,0))

@dataclass
class bucket:
    """
    Basic data class
    """
    pass

class FragmentOptions(BaseModel):
    functional : str = 'lda'
    fractional : bool = False
    interaction_type : str = 'dft'

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
        wfn.iterations(vp_matrix=potential)
    else:
        wfn.iterations()
    wfn.finalize_energy()

    basis_set = wfn.basisset()
    mints = psi4.core.MintsHelper(basis_set)
    T = mints.ao_kinetic()
    V = mints.ao_potential()


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
    frag_info.T        = T.np
    frag_info.V        = V.np
    frag_info.Ca_occ   = wfn.Ca_subset("AO", "OCC").np
    frag_info.Cb_occ   = wfn.Cb_subset("AO", "OCC").np
    frag_info.Ca_vir   = wfn.Ca_subset("AO", "VIR").np
    frag_info.Cb_vir   = wfn.Cb_subset("AO", "VIR").np
    frag_info.eig_a    = wfn.epsilon_a().np
    frag_info.eig_b    = wfn.epsilon_b().np
    frag_info.energies = energies
    frag_info.energy   = wfn.get_energies('Total Energy')

    return frag_info

class Fragment():
    """
    Initializes fragment. Hybrid between Psi4 Molecule adapted to mutliprocessing
    """

    # def __repr__(self):
    #     return self.mol_str

    def __init__(self, mol_str, basis, method, ref=1, plotting_grid='fine', optFragment={}):

        # Validate options
        optFragment = {k.lower(): v for k, v in optFragment.items()}
        for i in optFragment.keys():
            if i not in FragmentOptions().dict().keys():
                raise ValueError(f"{i} is not a valid option for a Fragment")
        optFragment = FragmentOptions(**optFragment)
        self.optFragment = optFragment

        # Identity of molecule
        self.mol_str   = mol_str
        self.basis_str = basis
        self.mol       = psi4.geometry(self.mol_str)
        self.method    = method
        self.ref = ref

        self.V = V() # Potential data bucket
        self.E = E() # Energy data bucket
        self.Plotter = Plotter() # Quantites on grid bucket
        self.Vnm = Vnm_fragment() # Quantities on ao basis set

        # ---_> Machinery to compute the scf

        # Build Psi4 basis object
        basis = psi4.core.BasisSet.build( self.mol, key='BASIS', target=self.basis_str)
        self.basis = basis
        self.nbf = self.basis.nbf()
        self.generate_jk()

        # Grid
        self.grid = Grider(self.mol_str, self.basis_str, self.ref, plotting_grid)

        # ----> PDFT 
        # Post SCF
        self.da = None
        self.db = None
        self.u = None #Chemical potential
        
        # PDFT
        self.Q = np.empty((self.grid.npoints))
        self.Alpha = None
        self.Beta  = None
        
        # Ensemble
        self.scale = 1.0
        self.calc_nuclear_potential()


    # ----> Methods

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

    def calc_nuclear_potential(self):        
        """
        Calculate external nuclear potential
        """
        print("\tCalculating External Potential")

        # Grid
        vext = self.grid.esp( Da=np.zeros((self.nbf, self.nbf)), 
                              Db=np.zeros((self.nbf, self.nbf)), 
                              vpot=self.grid.vpot, compute_hartree=False)
        self.V.vnuc       = vext

        # Plotting Grid
        # if self.plot_things:
        plot_vext = self.grid.esp(  Da=np.zeros((self.nbf, self.nbf)), 
                                    Db=np.zeros((self.nbf, self.nbf)), 
                                    grid=self.grid.plot_points, compute_hartree=False)
        self.Plotter.vnuc = plot_vext


    def calc_hxc_potential(self):
        """
        Calculates Hartree Exchange Correlation Potentials on the Grid
        """

        if self.optFragment.interaction_type == 'ni':
            self.vhxc = np.zeros_like(self.V.vnuc)

        if self.optFragment.interaction_type == 'dft':

            # # Hartree External on DFT Grid
            # if self.ref == 2:
            #     self.V.vh         = np.zeros((2,self.grid.npoints))
            #     self.V.vnuc       = np.zeros((2,self.grid.npoints))
            #     self.Plotter.vh   = np.zeros((2, self.grid.plot_npoints)) 
            #     self.Plotter.vnuc = np.zeros((2, self.grid.plot_npoints))

            #     self.V.vnuc[0,:], self.V.vh[0,:] = self.grid.esp( self.da, np.zeros_like(self.db), self.grid.vpot )
            #     self.V.vnuc[1,:], self.V.vh[1,:] = self.grid.esp( self.db, np.zeros_like(self.da), self.grid.vpot )
                
            #     # Plotting grid
            #     self.Plotter.vnuc[0,:], self.Plotter.vh[0,:] = self.grid.esp( self.da, np.zeros_like(self.db), grid=self.grid.plot_points )
            #     self.Plotter.vnuc[1,:], self.Plotter.vh[1,:] = self.grid.esp( self.db, np.zeros_like(self.db), grid=self.grid.plot_points )
            

            # else:
            self.V.vnuc, self.V.vh = self.grid.esp(self.da, self.db, self.grid.vpot) 

            if self.plot_things:
                self.Plotter.vnuc, self.Plotter.vh = self.grid.esp(self.da, self.db, grid=self.grid.plot_points)

            # Testing wether forming a single density from adding different alpha/beta turns fine.             
            # total_d = (self.da + self.db) / 2 
            # zeros   = np.zeros_like(total_d)
            # self.V.vx       = self.grid.vxc(func_id=1 , Da=total_d, Db=zeros, vpot=self.grid.vpot).T
            # self.V.vc       = self.grid.vxc(func_id=12, Da=total_d, Db=zeros, vpot=self.grid.vpot).T
            # self.Plotter.vx = self.grid.vxc(func_id=1 , Da=total_d, Db=zeros, grid=self.grid.plot_points).T
            # self.Plotter.vc = self.grid.vxc(func_id=12, Da=total_d, Db=zeros, grid=self.grid.plot_points).T

            # Correct Separate component
            # Exchange/Correlation on DFT grid
            self.V.vx       = self.grid.vxc(func_id=1 , Da=self.da, Db=self.db, vpot=self.grid.vpot).T
            self.V.vc       = self.grid.vxc(func_id=12, Da=self.da, Db=self.db, vpot=self.grid.vpot).T

            if self.plot_things:
                self.Plotter.vx = self.grid.vxc(func_id=1 , Da=self.da, Db=self.db, grid=self.grid.plot_points).T
                self.Plotter.vc = self.grid.vxc(func_id=12, Da=self.da, Db=self.db, grid=self.grid.plot_points).T
                if self.ref == 2:
                    self.Plotter.vx = np.sum( self.Plotter.vx, axis=0 ) /2
                    self.Plotter.vc = np.sum( self.Plotter.vc, axis=0 ) /2

            if self.ref == 2:
                self.V.vx       = np.sum( self.V.vx, axis=0 ) /2
                self.V.vc       = np.sum( self.V.vc, axis=0 ) /2

            self.V.vxc        = self.V.vx + self.V.vc
            self.V.vhxc       = self.V.vh + self.V.vx + self.V.vc

            if self.plot_things:
                self.Plotter.vxc  = self.Plotter.vx + self.Plotter.vc
                self.Plotter.vhxc = self.Plotter.vh + self.Plotter.vx + self.Plotter.vc

    def diagonalize(self, matrix, ndocc):
        A = self.A
        Fp = A.dot(matrix).dot(A)
        eigvecs, Cp = np.linalg.eigh(Fp)
        C = A.dot(Cp)
        Cocc = C[:, :ndocc]
        D = contract('pi,qi->pq', Cocc, Cocc)
        return C, Cocc, D, eigvecs

    def scf_manual(self, maxiter=50, vext=None):
        #Initial guess
        F = self.V.Vnm + self.V.Tnm
        C, Cocc, D, eigvecs = self.diagonalize(F, self.nalpha)
        D_old = D.copy()

        for i in range(maxiter):

            F = self.V.Vnm + self.V.Tnm
            J = np.einsum('pqrs,rs->pq', self.I, D, optimize=True)    
            F += 2*J

            n = psi4.core.Matrix.from_array( [ D ] )
            self.wfn.V_potential().set_D( [n,n] )
            vxc_a = psi4.core.Matrix( self.nbf, self.nbf )
            vxc_b = psi4.core.Matrix( self.nbf, self.nbf )
            self.wfn.V_potential().compute_V([vxc_a, vxc_b])
            F +=  vxc_a

            if vext is not None:
                F += vext[0]

            C, Cocc, D, eigvecs = self.diagonalize(F, self.nalpha)

            DD = np.abs(np.sum(D-D_old))
            D_old = D
            print("SCF Difference", DD)

            self.da = D
            self.db = D
            self.ca = C
            self.cb = C
            self.cocca = Cocc
            self.coccb = Cocc
            self.eigs_a = eigvecs
            self.eigs_b = eigvecs

            if DD < 1e-6:
                break

    def scf(self, 
                 vext= None,
                 maxiter=50):

        # Clean Psi4 variables
        # psi4.core.clean()
        # psi4.core.clean_options()
        # psi4.core.clean_variables()
        psi4.set_options({"save_jk" : True})

        mol = psi4.geometry(self.mol_str)
        wfn_base = psi4.core.Wavefunction.build(mol, self.basis_str)
        wfn = psi4.proc.scf_wavefunction_factory(self.method, wfn_base, "UKS")
        wfn.initialize()


        if vext is not None:
            wfn.iterations(vp_matrix=vext)
        else:
            wfn.iterations()
        wfn.finalize_energy()

        basis_set = wfn.basisset()
        mints = psi4.core.MintsHelper(basis_set)
        T = mints.ao_kinetic()
        V = mints.ao_potential()

        # Allocate T and V inside fragment
        # Do if statement to just calculate once
        self.V.Vnm = np.array(V).copy()
        self.V.Tnm = np.array(T).copy()
        self.nalpha = wfn.nalpha()
        self.nbeta  = wfn.nbeta()

        # if potential is not None:
        #     exc = generate_exc( mol_string, basis, wfn.Da().np )

        # Store PostSCF Quantites
        self.da = np.array(wfn.Da()).copy()
        self.db = np.array(wfn.Db()).copy()
        # self.dt = self.da + self.db
        # self.da = self.dt/2
        # self.db = self.dt/2
        self.ca = np.array(wfn.Ca()).copy()
        self.cb = np.array(wfn.Cb()).copy()
        self.ccca = np.array(wfn.Ca_subset("AO", "OCC")).copy()
        self.occb = np.array(wfn.Cb_subset("AO", "OCC")).copy()
        self.vira = np.array(wfn.Ca_subset("AO", "VIR")).copy()
        self.virb = np.array(wfn.Cb_subset("AO", "VIR")).copy()
        self.eigs_a = np.array(wfn.epsilon_a()).copy()
        self.eigs_b = np.array(wfn.epsilon_b()).copy()

        
        # Store Energies
        self.E.Enuc = wfn.get_energies('Nuclear')      # Nuclear Repulsion Energy
        self.E.Ekin = np.sum( (self.da + self.db) * T )
        self.E.Eext = np.sum( (self.da + self.db) * V )
        # self.E.e1  = wfn.get_energies('One-Electron')  # Nuclear/External + Kinetic
        # self.E.Ts  = self.E.e1 - self.E.Enuc           # Kinetic
        self.E.Evha = wfn.get_energies('Two-Electron')   # Hartree/Exchange Energy
        self.E.Evxc = wfn.get_energies('XC')             # Exchange correlation energy
        self.E.Etot = wfn.get_energies("Total Energy")   # Total electronic (?)

        if vext is None:
            self.E.E0 = copy(self.E.Etot)                # Isolated fragments without vp

        # Potentials
        self.Vnm.T = np.array(T).copy()
        self.Vnm.V = np.array(V).copy()
        self.Vnm.Vxca = np.array(wfn.Va()).copy()
        self.Vnm.vxcb = np.array(wfn.Vb()).copy()
        if self.ref == 1:
            self.Vnm.vh = 2 * wfn.jk().J()[0].np
        else: 
            self.Vnm.vh = wfn.jk().J()[0].np + wfn.jk().J()[1].np

        self.wfn = wfn
