"""
pdft_scf.py
Performs pdf calculations modeled after CADMIUM
"""

"""
optSCF 
etol
maxiter
kinetic_type
interacting
avoid_loop
calc_type = vp
"""

import numpy as np
import psi4

def pdft_scf(self, maxiter, interacting=True):

    etol      = 1e-6
    maxiter   = maxiter
    verbose   = True
    kinetic_type = 'inversion'
    interacting = True
    alpha       = 0.3

    if interacting:
        print(f"SCF Calculation on interacting fragments")
    else:
        print("SCF Calculation on isolated fragments")

    if verbose:
     print("\tInitial Guess: Calculations for isolated fragments.\n")    

    # Initial Guess: Calculations for isolated fragments. 
    for ifrag in self.frags:
        psi4.core.clean()
        psi4.core.clean_variables()
        ifrag.scf()

    # Calculate chemical potential (?)
    # if ab_sym:               
    #     self.mirror_ab( iter_zero=True )
        
    self.calc_protomolecule()
    self.calc_Q()

    print("Initial Density Difference:", np.linalg.norm(self.molSCF.da - self.dfa)) 
  
    # ----> Initialize SCF
    dif                 = 10.0
    old_E               = 0.0
    old_df              = self.dfa + self.dfb
    old_da = self.dfa.copy()
    old_db = self.dfb.copy()
    iterations          = 1
    inversionfailures   = 0
    STOP                = False
    vp_full          = [np.zeros((self.nbf, self.nbf)), np.zeros((self.nbf, self.nbf))]
    self.vp_grid     = np.zeros((self.grid.plot_npoints))

    if verbose:
        print("\tBegin PDFT-SCF Iterations")

    while ( dif > etol 
            and iterations <= maxiter 
            and not STOP ):

        self.iteration = iterations

    # ----> Calculate V_Hxc
        if verbose and iterations == 1:
            print("\t\t\tCalculating HXC Potentials")
        for ifrag in self.frags:
            # vhxc_old = ifrag.V.vhxc
            ifrag.calc_hxc_potential()
            # ifrag.vxc = 1.0 * ifrag.vxc + 0.0 * ifrag.vhxc_old

    # ----> Calculate Partition Potential
        if not interacting:
            vp = np.zeros_like(self.dfa)
            for ifrag in self.frags:
                ifrag.V.vp = vp
        else:
            if verbose and iterations == 1:
                print("\t\t\tCalculating Partition Potential")
            vp = self.partition_potential()
            self.current_vp = vp.copy()

            # coeff = np.linalg.norm(self.molSCF.da - self.dfa)
            coeff = 0.25

            if self.ref == 1:
                vp_nm = self.dft_grid_to_fock(vp[0,:], self.grid.vpot)
                self.current_vp_nm = vp_nm.copy() 
                # vp_nm = self.Vnm.vp
                # vp_nm = psi4.core.Matrix.from_array(vp_nm)
                # self.vp_grid += coeff * vp[0,:]
                # vp_full[0] += coeff * vp_nm
                # vp_full[1] += coeff * vp_nm
                # vp_nm = vp_full
                # self.vp_full_grid = vp_grid
                vp_nm = [vp_nm]
            else:
                vp_a = self.dft_grid_to_fock(vp[0,:], self.grid.vpot)
                vp_b = self.dft_grid_to_fock(vp[1,:], self.grid.vpot)
                self.current_vp_a_nm = vp_a.copy()
                self.current_vp_b_nm = vp_b.copy()
                vp_nm = [vp_a, vp_b]

    # ----> Add effective potetnials to fragments
        psi4.set_options({ "DAMPING_PERCENTAGE" : 20, 
                           "DIIS" : True, 
                           "FAIL_ON_MAXITER" : False,
                        #    "GUESS" : "GWH",
                           "maxiter" : 100, 
                           "D_CONVERGENCE" : 1e-8,
                           "E_CONVERGENCE" : 1e-8})
        for ifrag in self.frags:
            psi4.core.clean()
            psi4.core.clean_variables()

            ifrag.scf(vext=vp_nm)
            # ifrag.scf_manual(vext=vp_nm)

            #Calculate chemical potential

        #Linear mixing


            #Get energy. Collect energies
        psi4.set_options({"DAMPING_PERCENTAGE" : 0, "DIIS" : True, "maxiter" : 100,})

    # ---> Check convergence

        # Build new system
        self.calc_protomolecule()
        self.calc_Q()

        self.dfa = (1-alpha) * old_da + alpha * self.dfa
        self.dfb = (1-alpha) * old_db + alpha * self.dfb
        self.df = self.dfa + self.dfb
        old_da = self.dfa.copy()
        old_db = self.dfb.copy()

        self.energy()

    
    
        # self.calc_all_energies()

        # Check convergence
        scf_diff = np.linalg.norm(  self.df - old_df  )
        target_diff = self.molSCF.da - self.dfa
        old_df = self.df

        # target_d = self.density(Da=self.molSCF.da,     vpot=self.grid.vpot)
        # das      = self.density(Da=self.dfa,           vpot=self.grid.vpot)


        print(f" SCF DD: {scf_diff} | Grid DD: {np.linalg.norm(target_diff)} | FragA E: {self.frags[0].E.Etot} | FragB E: {self.frags[1].E.Etot}")


        iterations += 1

