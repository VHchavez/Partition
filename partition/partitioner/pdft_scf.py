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

def pdft_scf(self):

    etol      = 1e-6
    maxiter   = 20 
    calc_type = 'vp'

    interacting = True
    if interacting:
        print(f"Begin SCF Calculation on interacting fragments")
    else:
        print("Begin SCF Calculation on isolated fragments")

    

    # Make an individual calculation for each fragment. Initial Guess
    for ifrag in self.frags:
        ifrag.scf()
    # Calculate chemical potential (?)

    self.calc_protomolecule()
    self.calc_Q()



    # ----> Initialize SCF
    dif                 = 10.0
    old_E               = 0.0
    old_nf              = self.nf
    iterations          = 1
    inversionfailures   = 0
    STOP                = False

    while ( dif > etol 
            and iterations >= maxiter 
            and not STOP ):

        if kinetic_type == 'inversion' and interacting:
            print("Warning: Check CADMIUM")

    # ----> Calculate V_Hxc
        for ifrag in self.frags:
            vhxc_old = ifrag.vhxc
            ifrag.calc_hxc_potential()
            ifrag.vxc = 1.0 * ifrag.vxc + 0.0 * ifrag.vhxc_old

            """
            calc_hxc_potential
            Obtains the exchange correlation potential 
            energy and potential. Need to decide if on the grid or not. 
            """

    # ----> Calculate Partition Potential
    if not interacting:
        vp = np.zeros_like(self.dfa)
        for ifrag in self.frags:
            ifrag.V.vp = vp
    else:
        vp = self.partition_potential()

    # ----> Add effective potetnials to fragments
    for ifrag in self.frags:
        ifrag.scf(vext=vp)

        #Calculate Density

        #Calculate chemical potential

        #Linear mixing

        #Get energy. Collect energies

    # ---> Check convergence

        # Build new system
        self.calculate_protomolecule()
        self.calc_Q()
        self.calc_all_energies()

        # Check convergence

