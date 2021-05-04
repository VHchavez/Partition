"""
partitioner.py
"""

import numpy as np
import psi4
from dataclasses import dataclass
from pydantic import validator, BaseModel
from dask.distributed import Client

from ..inverter import Inverter
from ..grid.grider import Grider
from ..fragment import Fragment
from .pdft_scf import pdft_scf

# Partition Methods
from .partition_potential import partition_potential
from .vp_kinetic import vp_kinetic
from .vp_hxc import vp_hxc
# from .util import get_from_grid, basis_to_grid #eval_vh

class PartitionerOptions(BaseModel):
    vp_type      : str = 'component'
    hxc_type     : str = 'exact'
    kinetic_type : str = 'inversion'
    hxc_type     : str = 'exact'
    inv_method   : str = 'wuyang'
    opt_method   : str = 'bfgs'      
    k_family     : str = 'gga'
    plotting_grid : str = 'fine'
    ke_func_id   : int = 5
    ke_param     : dict = {}
    verbose      : bool = True
    interacting  : bool = True

@dataclass
class bucket:
    pass
@dataclass
class V:    
    pass
@dataclass
class Plotter:    
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


class Partitioner(Grider):
    def __init__(self, basis, method_str, frags_str=[], mol_str=None, ref=1, optPart={}):

        # Validate options
        optPart = {k.lower(): v for k, v in optPart.items()}
        for i in optPart.keys():
            if i not in PartitionerOptions().dict().keys():
                raise ValueError(f"{i} is not a valid option for Partitioner")
        optPart = PartitionerOptions(**optPart)
        self.optPart = optPart
    
        self.basis_str  = basis
        self.mol_str    = mol_str
        self.method_str = method_str
        self.frags_str  = frags_str
        self.frags      = None
        self.ref        = ref
        self.nfrags     = len( frags_str )
        self.ens = False

        # Data buckets
        self.V = V()
        self.Plotter = Plotter()

        #Client for Paralellization on fragments
        # self.client     = Client()

        # Psi4 Stuff
        self.mol   = psi4.geometry(self.mol_str)
        self.basis = psi4.core.BasisSet.build( self.mol, key='BASIS', target=self.basis_str)
        self.nbf   = self.basis.nbf()
        
        # Grider & Plotting 
        self.grid = Grider(self.mol_str, self.basis_str, self.ref, self.optPart.plotting_grid)
        
        # Generate fragments
        self.generate_fragments(self.optPart.plotting_grid)
        self.calc_nuclear_potential()

        # Inverter 
        if mol_str is not None:
            self.inverter = Inverter(self.mol, self.basis, self.ref, self.frags, self.grid)

    # ----> Methods

    def generate_fragments(self, plotting_grid):
        """
        Generate instance of Fragment for each fragment string
        """

        self.frags = []
        for i in self.frags_str:
            self.frags.append( Fragment(i, self.basis_str, self.method_str, self.ref, plotting_grid) )

    def calc_protomolecule(self):
        """
        Calculate the protomolecular density
        """

        # Evaluate sum of fragment densities and weiging functions
        
        self.da_frac = np.zeros_like(self.frags[0].da)
        if self.ref == 2:
            self.db_frac = np.zeros_like(self.frags[0].da)

        # Spin flip (?)

        # Scale for ensemble
        for ifrag in self.frags:
            self.da_frac += ifrag.da * ifrag.scale
            if self.ref == 2:
                self.db_frac += ifrag.db * ifrag.scale

            if self.ens:
                print("Need to iterate over ensemble set of fragments")

        # Sum of fragment densities
        self.dfa = self.da_frac
        if self.ref == 1:
            self.dfb = self.da_frac.copy()
        else:
            self.dfb = self.db_frac.copy()

        self.df = self.dfa + self.dfb

    def calc_nuclear_potential(self):
        """
        Calculate external nuclear potential
        """

        vnuc = np.zeros((self.grid.npoints))
        plot_vnuc = np.zeros((self.grid.plot_npoints))

        for ifrag in self.frags:
            vnuc       += ifrag.V.vnuc.copy()
            plot_vnuc  += ifrag.Plotter.vnuc.copy()

        # Plotting Grid
        self.V.vnuc = vnuc.copy()
        self.Plotter.vnuc = plot_vnuc.copy()

    def calc_Q(self):
        """
        Calculates Q functions according to PDFT
        """

        np.seterr(divide='ignore', invalid='ignore')

        # Fragment density on the grid

        if self.ref == 1:
            df = self.grid.density(grid=None, Da=self.dfa, vpot=self.grid.vpot)
            df = 2 * df

            #Plotter
            df_plotter = self.grid.density(Da=self.dfa, grid=self.grid.plot_points)
            df_plotter = 2 * df_plotter
        else:
            df = self.grid.density(grid=None, Da=self.dfa, Db=self.dfb, vpot=self.grid.vpot)
            df = df[:, 0] + df[:, 1]

            #Plotter
            df_plotter = self.grid.density(Da=self.dfa, Db=self.dfb, grid=self.grid.plot_points)
            df_plotter = df_plotter[:,0] + df_plotter[:,1]

        self.Plotter.df = df_plotter

        for ifrag in self.frags:
            if self.ref == 1:
                d = self.grid.density(grid=None, Da=ifrag.da, vpot=self.grid.vpot)
                d = 2 * d
                d_plotter = self.grid.density(Da=ifrag.da, grid=self.grid.plot_points)
                d_plotter = 2 * d_plotter
            else:
                d = self.grid.density(grid=None, Da=ifrag.da, Db=ifrag.db, vpot=self.grid.vpot)
                d = d[:,0] + d[:,1]
                d_plotter = self.grid.density(Da=ifrag.da, Db=ifrag.db, grid=self.grid.plot_points)
                d_plotter = d_plotter[:,0] + d_plotter[:,1]

            ifrag.Plotter.d = d_plotter

            ifrag.Q = ifrag.scale * d / df
            ifrag.Plotter.Q = ifrag.scale * d_plotter / df_plotter
            # Need to verify that q functions are functional. 

    def scf(self, maxiter=1):
        pdft_scf(self, maxiter)

    # ----> Potential Methods
    def partition_potential(self):
        return partition_potential(self)

    def vp_kinetic(self):
        vp_kinetic(self)

    def vp_hxc(self):
        vp_hxc(self)
# -----------------------------> OLD PARTITION


    def generate_mints_matrices(self):
        mints = psi4.core.MintsHelper( self.basis )

        self.S = mints.ao_overlap().np
        A = mints.ao_overlap()
        A.power(-0.5, 1.e-14)
        self.A = A
        self.S3 = np.squeeze(mints.ao_3coverlap(self.basis,self.basis,self.basis))
        self.jk = None
                    
    # def generate_jk(self, K=True, memory=2.50e8):
    #     jk = psi4.core.JK.build(self.basis)
    #     jk.set_memory(int(memory)) #1GB
    #     jk.set_do_K(K)
    #     jk.initialize()

    #     self.jk = jk

    # def form_jk(self, C_occ_a, C_occ_b):
    #     if self.jk is None:
    #         self.generate_jk()

    #     # C_occ_a = psi4.core.Matrix.from_array(C_occ_a)
    #     # C_occ_b = psi4.core.Matrix.from_array(C_occ_b)

    #     self.jk.C_left_add(C_occ_a)
    #     self.jk.C_left_add(C_occ_b)
    #     self.jk.compute()
    #     self.jk.C_clear()

    #     Ja = self.jk.J()[0].np
    #     Jb = self.jk.J()[1].np
    #     J = [Ja, Jb]

    #     Ka = self.jk.K()[0].np
    #     Kb = self.jk.K()[1].np
    #     K = [Ka, Kb]

    #     return J, K

    def scf_mol(self):
        
        method = self.method_str
        psi4.set_options({"maxiter" : 100})
        ret = self.client.map( _scf, [self.mol_str], [method], [self.basis_str] )
        data = [i.result() for i in ret]
        self.mol = data[0]

    def scf_frags(self,

            vext   = None,
            evaluate = False):

        method = self.method_str

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
        coc_suma = np.zeros_like( self.frags[0].Ca_occ )
        coc_sumb = np.zeros_like( self.frags[0].Cb_occ )

        for i in range(self.nfrags):
            n_suma += self.frags[i].Da
            n_sumb += self.frags[i].Db
            coc_suma += self.frags[i].Ca_occ
            coc_sumb += self.frags[i].Cb_occ

        self.frags_na = n_suma.copy()
        self.frags_nb = n_sumb.copy()
        self.frags_coca = coc_suma.copy()
        self.frags_cocb = coc_sumb.copy()



