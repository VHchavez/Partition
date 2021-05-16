"""
partitioner.py
"""

from copy import copy 

import numpy as np
import psi4
from dataclasses import dataclass
from pydantic import validator, BaseModel
from dask.distributed import Client

from ..inverter import Inverter
from ..grid.grider import Grider
from ..fragment import Fragment
from .pdft_scf import pdft_scf
from .energy import energy
from .partition_energy import partition_energy

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
        if not self.ens:
            ifrag = self.frags
        else:
            pass
            #ifrag = self.frags self.efrags
            # union de fragmentos y efragmetnos

        # Initialize fractional densities
        for i in ifrag:
            i.da_frac = np.zeros_like( i.da )
            if self.ref == 2:
                i.db_frac = np.zeros_like( i.db )
            else:
                i.db_frac = i.da_frac.copy()

        # Spinflip (?)

        # Scale for ensemble
        for i in ifrag:
            i.da_frac += i.da * i.scale
            if self.ref == 2:
                i.db_frac += i.db * i.scale
            else:
                i.db_frac = i.da_frac.copy()

        # Sum of fragment densities
        self.dfa = np.zeros_like( self.frags[0].da )
        if self.ref == 2:
            self.dfb = np.zeros_like( self.frags[0].da )
        else:
            self.dfb = self.dfa.copy()

        for i in ifrag:
            self.dfa += i.da_frac
            if self.ref == 2:
                self.dfb += i.db_frac
            else:
                self.dfb = self.dfa.copy()

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

    # ----> Energy Methods
    def energy(self):
        """
        Gathers energies from all fragments
        """
        energy(self)

    def energy(self):
        """
        Calculates the partition energy of the system
        """
        partition_energy(self)
# -----------------------------> OLD PARTITION


    def generate_mints_matrices(self):
        mints = psi4.core.MintsHelper( self.basis )

        self.S = mints.ao_overlap().np
        A = mints.ao_overlap()
        A.power(-0.5, 1.e-14)
        self.A = A
        self.S3 = np.squeeze(mints.ao_3coverlap(self.basis,self.basis,self.basis))
        self.jk = None
                    
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

    def mirror_ab(self, iter_zero=False):
        ### Mirrors:
        # Density, Q, vx, vc, vh, vp
        x,y,z,_ = self.grid.vpot.get_np_xyzw()
        ive_been_flipped = [ False for i in range(len(x)) ]
        
        if self.ref == 1:
            density = self.grid.density(Da=self.frags[0].da, vpot=self.grid.vpot)
        else:
            density = self.grid.density(Da=self.frags[0].da, Db=self.frags[0].da, vpot=self.grid.vpot)

        flip_d  =  np.zeros_like(x)
        self.frags[1].Q     =  np.zeros_like(x)
        self.frags[1].V.vx  =  np.zeros_like(x)
        self.frags[1].V.vc  =  np.zeros_like(x)
        self.frags[1].V.vh  =  np.zeros_like(x)
        self.frags[1].V.vp  =  np.zeros_like(x)
        
        # Go through all points
        for i in range(len(x)):

            # Find the other point mirroring along the zaxis
            flipid = np.intersect1d( np.intersect1d( np.where(x == x[i])[0], np.where(y == y[i])[0]), np.where(z == -z[i])[0] )[0]

            # Replace both points
            if not ive_been_flipped[i]:

                flip_d[flipid] = copy(density[i])                          # Density
                flip_d[i]      = copy(density[flipid])

                if not iter_zero:
                    self.frags[1].Q[flipid]    = copy(self.frags[0].Q[i])       # Q
                    self.frags[1].Q[i]         = copy(self.frags[0].Q[flipid])

                    self.frags[1].V.vx[flipid] = copy(self.frags[0].vx[i])       # Exchange
                    self.frags[1].V.vx[i]      = copy(self.frags[0].vx[flipid])

                    self.frags[1].V.vc[flipid] = copy(self.frags[0].vc[i])       # Correlation
                    self.frags[1].V.vc[i]      = copy(self.frags[0].vc[flipid])

                    self.frags[1].V.vh[flipid] = copy(self.frags[0].vh[i])       # Hartree
                    self.frags[1].V.vh[i]      = copy(self.frags[0].vh[flipid])

                ive_been_flipped[i]      = True
                ive_been_flipped[flipid] = True

        self.density_zero = density.copy()
        self.density_inv  = flip_d.copy() 
                
        if self.ref == 1:
            self.frags[1].da = self.dft_grid_to_fock( flip_d, Vpot=self.grid.vpot )
            self.frags[1].db = self.frags[1].da.copy()
        else:
            self.frags[1].db = self.dft_grid_to_fock( flip_d, Vpot=self.grid.vpot )


