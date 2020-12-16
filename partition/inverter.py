"""
inverter.py
"""

import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt

from opt_einsum import contract

from pyscf import scf

from .util import to_grid, basis_to_grid, get_from_grid

import psi4
psi4.core.be_quiet()

import sys

class Inverter():

    def __init__(self, partition_object,
                       debug=False):

        self.part     = partition_object
        self.inv_type = "xc" if self.part.frags is None else "partition"
        self.opt_method = None

        self.nauxbf   = self.part.nbf

        self.grad_a = None
        self.grad_b = None

        #Target Quantities
        self.ct = None
        self.nt = None

        #Inverted Potential
        self.v = np.zeros( 2 * self.nauxbf )
        self.reg = 0.0

        self.debug  = debug

    def initial_guess(self, guess):
        """
        Provides initial guess for inversion
        """ 

        if self.inv_type == "xc":

            print("Generating Initial Guess")

            N = self.part.mol.nalpha + self.part.mol.nbeta
            self.guess_a, self.guess_b = np.zeros_like(self.part.T), np.zeros_like(self.part.T)
            self.guess_ar, self.guess_br = [], []
    
            if True: #If grid is True
                bucket = get_from_grid(self.part.mol_str, self.part.basis_str, self.nt[0], self.nt[1])

            self.grid = [bucket.x, bucket.y, bucket.z]
            self.zz      = bucket.zz
            self.vxc_a_tar = bucket.vxc_a
            self.vxc_b_tar = bucket.vxc_b
            self.vha_tar = bucket.vha
            self.vxc_az  = bucket.vxc_az
            self.vxc_bz  = bucket.vxc_bz
            self.vha_z   = bucket.vha_z

            self.guess_ra = np.zeros_like(self.zz)
            self.guess_rb = np.zeros_like(self.zz)

            if "none"         in [i.lower() for i in guess]:
                self.guess_a +=   np.zeros_like(self.part.T) 
                self.guess_a +=   np.zeros_like(self.part.T) 

            if "fermi_amaldi" in [i.lower() for i in guess]:
                print("Adding Fermi Amaldi potential to guess")
                J, _ = self.part.form_jk( self.ct[0], self.ct[1] )
                v_fa =  (-1.0 / N) * (J[0] + J[1])

                self.guess_a += v_fa.copy()
                self.guess_b += v_fa.copy()
                self.guess_ra += self.vha_z
                self.guess_rb += self.vha_z


            if "svwn"          in [i.lower() for i in guess]:

                print("Adding XC potential to initial guess")

                psi4.set_options( { "reference" : psi4.core.get_global_option("REFERENCE") } )
                mol_guess = psi4.geometry(self.part.mol_str)
                _, wfn_guess = psi4.energy("svwn"+"/"+self.part.basis_str, molecule=mol_guess, return_wfn=True)

                na_target = psi4.core.Matrix.from_array( self.nt[0] )
                nb_target = psi4.core.Matrix.from_array( self.nt[1] )

                #Get VXC using exact density matrix
                wfn_guess.V_potential().set_D([ na_target, nb_target ])
                #wfn_guess.V_potential().properties()[0].set_pointers( na_target, nb_target ) #In order to get it to the grid points
                va_target = psi4.core.Matrix.from_array( np.zeros_like( self.nt[0] ) )
                vb_target = psi4.core.Matrix.from_array( np.zeros_like( self.nt[1] ) )
                wfn_guess.V_potential().compute_V([va_target, vb_target]) 

                self.guess_a += va_target.np.copy()
                self.guess_b += vb_target.np.copy()
                self.guess_ra += self.vxc_az
                self.guess_rb += self.vxc_bz

                if self.debug == True:
                    print("Hartree A\n")
                    print(J[0][:2,:2] + J[1][:2,:2])
                    print("\n")

                    print("Vxc with exact density\n")
                    print(va_target.np[:2,:2])
                    print("\n")

                    print("External Potential\n")
                    print("V\n",self.part.V[:2,:2])

                    print("Kinetic\n")
                    print(self.part.T[:2,:2])

                    print("Guess\n")
                    print(self.guess_a[:2,:2])


        else:
            if guess.lower() == "none":
                self.guess_a = np.zeros_like(self.part.V)
                self.guess_b = np.zeros_like(self.part.V)

            elif guess.lower() == "core_xc":

                self.guess_a = self.part.T + self.part.V + self.part.mol.Va
                self.guess_b = self.part.T + self.part.V + self.part.mol.Vb

                for i in range(self.part.nfrags):
                    self.guess_a -= self.part.Ts[i] + self.part.Vs[i] + self.part.frags[i].Va
                    self.guess_b -= self.part.Ts[i] + self.part.Vs[i] + self.part.frags[i].Vb


    def lagrangian(self, v):
        vks_a = 1.0 * np.einsum("ijk,k->ij", self.part.S3, v[:self.part.nbf]) + self.guess_a
        vks_b = 1.0 * np.einsum("ijk,k->ij", self.part.S3, v[self.part.nbf:]) + self.guess_b
        fock_a = self.part.V + self.part.T + vks_a 
        fock_b = self.part.V + self.part.T + vks_b

        self.Ca, self.Coca, self.Da, self.eigvecs_a = self.build_orbitals( fock_a, self.part.mol.nalpha )
        self.Cb, self.Cocb, self.Db, self.eigvecs_b = self.build_orbitals( fock_b, self.part.mol.nbeta )

        if False:  #Plot potential at each step
            pb, v_guess = self.part.generate_1D_phi( self.nt[0] + self.nt[1], "pbe")

            vgrida = np.einsum('t,rt->r', v[:self.part.nbf], pb)
            vgridb = np.einsum('t,rt->r', v[self.part.nbf:], pb)

            plt.plot(self.part.grid[:,0], vgrida + vgridb, label="From Optimizer")
            plt.plot(self.part.grid[:,0], v_guess, label="Initial Guess")
            plt.plot(self.part.grid[:,0], v_guess + vgrida + vgridb, label="V_ks")
            plt.legend()

            
            #USING PSI4 UNIQUELY
            # vgrid = to_grid(v[:self.part.nbf], vpot=self.vpot)
            # vgrid, grid = basis_to_grid( v[:self.part.nbf], Vpot=self.vpot, blocks=False )
            # self.xyz_v = vgrid
            # self.xyz   = grid
            # plt.plot(vgrid)

            plt.show()

        self.grad_a = np.einsum('ij,ijt->t', (self.Da - self.nt[0]), self.part.S3)
        self.grad_b = np.einsum('ij,ijt->t', (self.Db - self.nt[1]), self.part.S3) 

        kinetic    = np.einsum('ij,ji', self.part.T, (self.Da + self.Db))
        potential  = np.einsum('ij,ji', self.part.V, (self.Da + self.Db) - (self.nt[0] + self.nt[1]))
        potential += np.einsum('ij,ji', self.guess_a, (self.Da - self.nt[0]) )
        potential += np.einsum('ij,ji', self.guess_b, (self.Db - self.nt[1]) )
        optz       = np.einsum('i,i'  , v[:self.part.nbf], self.grad_a)
        optz      += np.einsum('i,i'  , v[self.part.nbf:], self.grad_b)
    
        print(f" L1: {kinetic}, L2: {potential}, L3: {optz}")

        L = kinetic + potential + optz

        penalty = 0.0
        if self.reg > 0:
            penalty = self.reg * self.Dvb(v)

        return -(L - penalty)

    def gradient(self, v):
        vks_a = 1.0 *np.einsum("ijk,k->ij", self.part.S3, v[:self.part.nbf]) + self.guess_a
        vks_b = 1.0 *np.einsum("ijk,k->ij", self.part.S3, v[self.part.nbf:]) + self.guess_b
        fock_a = self.part.V + self.part.T + vks_a 
        fock_b = self.part.V + self.part.T + vks_b

        self.Ca, self.Coca, self.Da, self.eigvecs_a = self.build_orbitals( fock_a, self.part.mol.nalpha )
        self.Cb, self.Cocb, self.Db, self.eigvecs_b = self.build_orbitals( fock_b, self.part.mol.nalpha )

        self.grad_a = contract('ij,ijt->t', (self.Da - self.nt[0]), self.part.S3)
        self.grad_b = contract('ij,ijt->t', (self.Db - self.nt[1]), self.part.S3) 

        if self.reg > 0:
            self.grad_a += 2 * self.reg * contract('st,t->s', self.part.T, v[:self.nauxbf])
            self.grad_b += 2 * self.reg * contract('st,t->s', self.part.T, v[self.nauxbf:])

        self.grad = np.concatenate( (self.grad_a, self.grad_b) )

        return -self.grad

    def hessian(self, v):
        vks_a = 1.0 * np.einsum("ijk,k->ij", self.part.S3, v[:self.part.nbf]) + self.guess_a
        vks_b = 1.0 * np.einsum("ijk,k->ij", self.part.S3, v[self.part.nbf:]) + self.guess_b
        fock_a = self.part.V + self.part.T + vks_a 
        fock_b = self.part.V + self.part.T + vks_b

        Ca, Coca, Da, eigvecs_a = self.build_orbitals( fock_a, self.part.mol.nalpha )
        Cb, Cocb, Db, eigvecs_b = self.build_orbitals( fock_b, self.part.mol.nalpha )

        nalpha = self.part.mol.nalpha
        nbeta  = self.part.mol.nbeta
        Ca = self.Ca
        Cb = self.Cb
        self.Coca = Coca
        self.Cocb = Cocb
        eigs_diff_a = self.eig_a[:nalpha, None] - self.eig_a[None,nalpha:] 
        eigs_diff_b = self.eig_b[:nbeta,  None] - self.eig_b[None,nbeta: ]

        C3a = np.einsum('mi,va,mvt->iat', Ca[:,:nalpha], Ca[:,nalpha:], self.part.S3)
        C3b = np.einsum('mi,va,mvt->iat', Cb[:,:nbeta ], Cb[:,nbeta: ], self.part.S3)

        Ha = 2 * np.einsum('iau,iat,ia->ut', C3a, C3a, eigs_diff_a**-1)
        Hb = 2 * np.einsum('iau,iat,ia->ut', C3b, C3b, eigs_diff_b**-1)

        Hs = np.block( 
                        [[Ha, np.zeros((self.nauxbf, self.nauxbf))],
                         [np.zeros((self.nauxbf, self.nauxbf)), Hb]] 
                     )

        if self.reg > 0:
            Hs[self.nauxbf:,self.nauxbf:] -= 2*self.reg*self.part.T
            Hs[:self.nauxbf,:self.nauxbf] -= 2*self.reg*self.part.T

        return -Hs

    def build_orbitals(self, diag, ndocc):

        diag = psi4.core.Matrix.from_array( diag )
        Fp = psi4.core.triplet(self.part.A, diag, self.part.A, True, False, True)
        nbf = self.part.A.shape[0]
        Cp = psi4.core.Matrix(nbf, nbf)
        eigvecs = psi4.core.Vector(nbf)
        Fp.diagonalize(Cp, eigvecs, psi4.core.DiagonalizeOrder.Ascending)
        C = psi4.core.doublet(self.part.A, Cp, False, False)
        Cocc = psi4.core.Matrix(nbf, ndocc)
        Cocc.np[:] = C.np[:, :ndocc]
        D = psi4.core.doublet(Cocc, Cocc, False, True)

        C       = C.np
        Cocc    = Cocc.np
        D       = D.np
        eigvecs = eigvecs.np

        return C, Cocc, D, eigvecs

    def finalize_energy(self):

        coca = psi4.core.Matrix.from_array(self.Coca)
        cocb = psi4.core.Matrix.from_array(self.Cocb)

        J, K = self.part.form_jk( coca, cocb )

        energy_kinetic    = contract('ij,ij', self.part.T, (self.Da + self.Db))
        energy_external   = contract('ij,ij', self.part.V, (self.Da + self.Db))
        energy_hartree_a  = 0.5 * contract('ij,ji', J[0] + J[1], self.Da)
        energy_hartree_b  = 0.5 * contract('ij,ji', J[0] + J[1], self.Db)

        print("WARNING: XC Energy is not yet properly calculated")

        # alpha = 0.0
        bucket = get_from_grid(self.part.mol_str, self.part.basis_str, self.Da, self.Db )
        # energy_exchange_a = -0.5 * alpha * contract('ij,ji', K[0], self.Da)
        # energy_exchange_b = -0.5 * alpha * contract('ij,ji', K[1], self.Db)
        energy_ks            =  1.0 * bucket.exc

        energies = {"One-Electron Energy" : energy_kinetic + energy_external,
                    "Two-Electron Energy" : energy_hartree_a + energy_hartree_b,
                    "XC"                  : energy_ks,
                    "Total Energy"        : energy_kinetic   + energy_external  + \
                                            energy_hartree_a + energy_hartree_b + \
                                            energy_ks }

        self.energies = energies

    def density_accuracy(self):

        mol_string = self.part.mol_str

        n_ra = self.Da
        n_rb = self.Db
        nt_ra = self.nt[0]
        nt_rb = self.nt[0]

        density_difference = 0.0
        
        mol_grid = psi4.geometry(mol_string)
        _, wfn = psi4.energy( "svwn/"+self.part.basis_str, molecule=mol_grid, return_wfn=True)

        vpot = wfn.V_potential()
        points = vpot.properties()[0]
        functional = vpot.functional()

        xc_e = 0.0

        for b in range(vpot.nblocks()):

            block = vpot.get_block(b)
            points.compute_points(block)
            npoints = block.npoints()
            lpos = np.array( block.functions_local_to_global() )
            
            w = np.array(block.w())
            phi = np.array( points.basis_values()["PHI"])[:npoints, :lpos.shape[0]]

            lDa  = n_ra[(lpos[:, None], lpos)]
            lDb  = n_rb[(lpos[:, None], lpos)]
            lDta = nt_ra[(lpos[:, None], lpos)]
            lDtb = nt_rb[(lpos[:, None], lpos)]
            rho_a = np.einsum('pm,mn,pn->p', phi, lDa, phi, optimize=True)
            rho_b = np.einsum('pm,mn,pn->p', phi, lDb, phi, optimize=True) 
            rho_ta = np.einsum('pm,mn,pn->p', phi, lDta, phi, optimize=True)
            rho_tb = np.einsum('pm,mn,pn->p', phi, lDtb, phi, optimize=True)

            density_difference += np.einsum('a,a', w, np.abs(rho_a + rho_b - rho_ta - rho_tb ), optimize=True)

        print(f"Density Accuracy: {density_difference}")

    def Dvb(self, v=None):
        "From KSPIES "
        if v is None:
            va = self.v[:len(self.v)//2]
            vb = self.v[len(self.v)//2:]
        else:
            va = v[:len(v)//2]
            vb = v[len(v)//2:]
        Dvb  = np.einsum('s,st,t', va, self.part.T, va)
        Dvb += np.einsum('s,st,t', vb, self.part.T, vb)
        return Dvb

    def invert(self, wfn,
                     opt_method='trust-krylov', 
                     v_guess="swvn"):

        self.nt = [wfn.Da().np, wfn.Db().np]
        self.ct = [wfn.Ca_subset("AO", "OCC"), wfn.Cb_subset("AO", "OCC")]
        self.initial_guess(v_guess)
        self.opt_method = opt_method

        if self.debug == True:
            print("Optimizing")
    
        if self.opt_method.lower() == 'bfgs':
            opt_results = minimize( fun = self.lagrangian,
                                    x0  = self.v, 
                                    jac = self.gradient,
                                    method = self.opt_method,
                                    # tol    = 1e-10,
                                    options = {"maxiter" : 10,
                                               "disp"    : False,}
                                    )

        else:
            opt_results = minimize( fun = self.lagrangian,
                                    x0  = self.v, 
                                    jac = self.gradient,
                                    hess = self.hessian,
                                    method = self.opt_method,
                                    # tol    = 1e-10,
                                    options = {"maxiter" : 10,
                                               "disp"    : False, }
                                    )

        print(opt_results.message)

        if opt_results.success is False:
            raise ValueError("Optimization was unsucessful, try a different intitial guess")
        
        self.finalize_energy()
        self.density_accuracy()


        self.v = opt_results.x
             

    #PDFT INVERSION ###################################################################################

    def pdft_invert(self, target_density, 
                     target_cocc=None, 
                     opt_method='trust-krylov', 
                     v_guess="core"):

        self.ct = target_cocc
        self.nt = target_density
        print("I am about to calculate initial guess")
        self.initial_guess(v_guess)
        self.opt_method = opt_method

        if self.debug == True:
            print("Optimizing")
    
        if self.opt_method.lower() == 'bfgs':
            opt_results = minimize( fun = self.pdft_lagrangian,
                                    x0  = self.v, 
                                    jac = self.pdft_gradient,
                                    method = self.opt_method,
                                    # tol    = 1e-10,
                                    options = {"maxiter" : 1000,
                                               "disp"    : True,}
                                    )

        else:
            opt_results = minimize( fun = self.pdft_lagrangian,
                                    x0  = self.v, 
                                    jac = self.pdft_gradient,
                                    hess = self.pdft_hessian,
                                    method = self.opt_method,
                                    # tol    = 1e-10,
                                    options = {"maxiter" : 1000,
                                               "disp"    : True, }
                                    )

        # if opt_results.success is False:
        #     raise ValueError("Optimization was unsucessful, try a different intitial guess")
        
        print("Finalized Optimization Successfully")
        print(opt_results.message)

        self.pdft_finalize_energy()
        self.v = opt_results.x

    def pdft_lagrangian(self, v):
        vp_a = np.einsum("ijk,k->ij", self.part.S3, v[:self.part.nbf]) + self.guess_a
        vp_b = np.einsum("ijk,k->ij", self.part.S3, v[self.part.nbf:]) + self.guess_b

        self.part.scf_frags(method="svwn", vext=(vp_a, vp_b))

        if False:  #Plot potential at each step
            
            atom = """He 0.0 0.0 -0.4
                     He 0.0 0.0 0.4"""

            pb, v_guess = self.part.generate_1D_phi( atom, self.nt[0] + self.nt[1], "pbe")

            vgrida = np.einsum('t,rt->r', v[:self.part.nbf], pb)
            vgridb = np.einsum('t,rt->r', v[self.part.nbf:], pb)


            plt.plot(self.part.grid[:,0], vgrida + vgridb, label="From Optimizer")
            # plt.plot(self.part.grid[:,0], vgridb)
            # plt.plot(self.part.grid[:,0], v_guess, label="Initial Guess")
            # plt.plot(self.part.grid[:,0], v_guess + vgrida + vgridb, label="Vp")

            # plt.xlim(0, 3.0)

            # plt.xscale('log')

            plt.legend()

            
            #USING PSI4 UNIQUELY
            # vgrid = to_grid(v[:self.part.nbf], vpot=self.vpot)
            # vgrid, grid = basis_to_grid( v[:self.part.nbf], Vpot=self.vpot, blocks=False )
            # self.xyz_v = vgrid
            # self.xyz   = grid
            # plt.plot(vgrid)

            plt.show()

        self.part.frag_sum()

        Da = self.part.frags_na
        Db = self.part.frags_nb

        self.grad_a = np.einsum('ij,ijt->t', (Da-self.nt[0]), self.part.S3)
        self.grad_b = np.einsum('ij,ijt->t', (Db-self.nt[1]), self.part.S3) 

        print("Gradient Convergence", np.linalg.norm( self.grad_a + self.grad_b ))

        kinetic = 0.0
        potential = 0.0
        optz      = 0.0 

        for i in range(self.part.nfrags):
            Da = self.part.frags[i].Da
            Db = self.part.frags[i].Db
            T  = self.part.Ts[i]
            V  = self.part.Vs[i]

            kinetic   += np.einsum('ij,ji', self.part.T, (Da + Db))
            potential += np.einsum('ij,ji', self.part.V, (Da + Db))
            potential += self.part.frags[i].energies["e2"]
            potential += self.part.frags[i].energies["exc"]

        optz      += np.einsum('t,t', v[:self.nauxbf], self.grad_a)
        optz      += np.einsum('t,t', v[self.nauxbf:], self.grad_b)

        print(f" L1: {kinetic}, L2: {potential}, L3: {optz}")

        L = kinetic + potential + optz

        penalty = 0.0
        if self.reg > 0:
            penalty = self.reg * self.Dvb(v)

        return -(L - penalty)

    def pdft_gradient(self, v):

        vp_a = np.einsum("ijk,k->ij", self.part.S3, v[:self.part.nbf]) + self.guess_a
        vp_b = np.einsum("ijk,k->ij", self.part.S3, v[self.part.nbf:]) + self.guess_b

        self.part.scf_frags(method="svwn", vext=[vp_a, vp_b])

        self.part.frag_sum()
        Da = self.part.frags_na
        Db = self.part.frags_nb

        self.grad_a = contract('ij,ijt->t', (Da-self.nt[0]), self.part.S3)
        self.grad_b = contract('ij,ijt->t', (Db-self.nt[1]), self.part.S3) 

        if self.reg > 0:
            self.grad_a = 2 * self.reg * contract('st,t->s', self.part.T, v[:self.nauxbf])
            self.grad_b = 2 * self.reg * contract('st,t->s', self.part.T, v[self.nauxbf:])

        self.grad = np.concatenate( (self.grad_a, self.grad_b) )

        return -self.grad

    def pdft_finalize_energy(self):

        pass

        # self.part.frag_sum()

        # Coca = self.part.frags_coca
        # Cocb = self.part.frags_coca
        # Da = self.part.frags_na
        # Db = self.part.frags_nb

        # J, K = self.part.form_jk( Coca, Cocb )
        # ks_e  = generate_exc( self.part.frags[0].mol_str, self.part.basis_str, Da, Db )

        # energy_kinetic    = contract('ij,ij', self.part.T, (Da + Db))
        # energy_external   = contract('ij,ij', self.part.V, (Da + Db))
        # energy_hartree_a  = 0.5 * contract('ij,ji', J[0] + J[1], Da)
        # energy_hartree_b  = 0.5 * contract('ij,ji', J[0] + J[1], Db)
        # energy_ks            =  1.0 * ks_e

        # energies = {"One-Electron Energy" : energy_kinetic + energy_external,
        #             "Two-Electron Energy" : energy_hartree_a + energy_hartree_b,
        #             "XC"      : ks_e,
        #             "Total Energy" : energy_kinetic   + energy_external  +  \
        #                              energy_hartree_a + energy_hartree_b + \
        #                              ks_e}

        # self.energies = energies

        