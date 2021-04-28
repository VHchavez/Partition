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

def pdft_scf(self, maxiter, interacting=True):

    etol      = 1e-6
    maxiter   = maxiter
    calc_type = 'vp'
    verbose   = True
    kinetic_type = 'inversion'
    interacting = True

    if interacting:
        print(f"Begin SCF Calculation on interacting fragments")
    else:
        print("Begin SCF Calculation on isolated fragments")

    if verbose:
     print("Preparing Initial Guess \n")    

    # Make an individual calculation for each fragment. Initial Guess
    for ifrag in self.frags:
        ifrag.scf()
    # Calculate chemical potential (?)
    self.calc_protomolecule()
    self.calc_Q()

    # ----> Initialize SCF
    dif                 = 10.0
    old_E               = 0.0
    old_nf              = self.dfa + self.dfb
    iterations          = 1
    inversionfailures   = 0
    STOP                = False

    if verbose:
        print("Begin SCF Iterations\n")

    while ( dif > etol 
            and iterations <= maxiter 
            and not STOP ):

        # # Sanity Check
        # if kinetic_type == 'inversion' and interacting:
        #     if iterations == 1:
        #         avoid_loop = True
        #     else:
        #         avoid_loop False 

    # ----> Calculate V_Hxc
        if verbose and iterations == 1:
            print("Calculating HXC Potentials")
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
                print("Calculating Partition Potential")
            vp = self.partition_potential()
            vp_nm = self.dft_grid_to_fock(vp, self.grid.vpot)

    # ----> Add effective potetnials to fragments
        for ifrag in self.frags:
            ifrag.scf(vext=vp_nm)
            #Calculate chemical potential

            #Linear mixing

            #Get energy. Collect energies
            #ifrag.energy()

    # ---> Check convergence

        # Build new system
        self.calc_protomolecule()
        self.calc_Q()
        # self.calc_all_energies()

        # Check convergence
        dif_df = np.max(  self.df - old_nf  )

        print(f" SCF Density Difference: {dif_df} | FragA Energy: {self.frags[0].et} | FragB Energy: {self.frags[1].et}")

        iterations += 1

