
"""
partition_potential.py
"""

import numpy as np

def partition_potential(self):
    """
    Calculate the partition potential
    """

    if self.optPart.vp_type == "component":

        # if self.optPart.verbose:
        #     print("Calculating the components of vp")
        # target_a = self.molSCF.da
        # target_b = self.molSCF.db
        target_a = self.dfa
        target_b = self.dfb

        self.inverter.dt = [target_a, target_b]
        self.inverter.ct = []
        self.inverter.eigvecs_a = []
        self.inverter.eigvecs_b = []

        # Calculate Nuclear Component
        for ifrag in self.frags:
            ifrag.V.vp_pot       = self.V.vnuc - ifrag.V.vnuc
            ifrag.Plotter.vp_pot = self.Plotter.vnuc - ifrag.Plotter.vnuc
            ifrag.Vnm.vp_pot     = self.Vnm.V - ifrag.Vnm.V

        # Calculate Kinetic Component
        self.vp_kinetic()

        # Calcualte HXC Component
        self.vp_hxc()

        # Build the partition potential and components
        self.V.vp     = np.zeros((self.ref,self.grid.npoints))
        self.V.vp_pot = np.zeros((self.ref,self.grid.npoints))
        self.V.vp_kin = np.zeros((self.ref,self.grid.npoints))
        self.V.vp_hxc = np.zeros((self.ref,self.grid.npoints))
        self.V.vp_h   = np.zeros((self.ref,self.grid.npoints))
        self.V.vp_x   = np.zeros((self.ref,self.grid.npoints))
        self.V.vp_c   = np.zeros((self.ref,self.grid.npoints))
        
        # Vnm
        self.Vnm.vp     = np.zeros((self.nbf, self.nbf))
        self.Vnm.vp_pot = np.zeros((self.nbf, self.nbf))
        self.Vnm.vp_kin = np.zeros((self.nbf, self.nbf))
        self.Vnm.vp_hxc = np.zeros((self.nbf, self.nbf))
        self.Vnm.vp_h   = np.zeros((self.nbf, self.nbf))
        self.Vnm.vp_xc  = np.zeros((self.nbf, self.nbf))

        # if converged == True
        if self.plot_things:
            self.Plotter.vp     = np.zeros((self.ref,self.grid.plot_npoints))
            self.Plotter.vp_kin = np.zeros((self.ref,self.grid.plot_npoints))
            self.Plotter.vp_pot = np.zeros((self.ref,self.grid.plot_npoints))
            self.Plotter.vp_hxc = np.zeros((self.ref,self.grid.plot_npoints))
            self.Plotter.vp_h = np.zeros((self.ref,self.grid.plot_npoints))
            self.Plotter.vp_x = np.zeros((self.ref,self.grid.plot_npoints))
            self.Plotter.vp_c = np.zeros((self.ref,self.grid.plot_npoints))

        for ifrag in self.frags:

            # DFT Grid
            ifrag.V.vp       = ifrag.V.vp_pot + ifrag.V.vp_kin + ifrag.V.vp_hxc
            self.V.vp     += ifrag.V.vp     * ifrag.Q
            self.V.vp_pot += ifrag.V.vp_pot * ifrag.Q
            self.V.vp_kin += ifrag.V.vp_kin * ifrag.Q
            self.V.vp_hxc += ifrag.V.vp_hxc * ifrag.Q
            self.V.vp_h   += ifrag.V.vp_h   * ifrag.Q
            self.V.vp_x   += ifrag.V.vp_x   * ifrag.Q
            self.V.vp_c   += ifrag.V.vp_c   * ifrag.Q

            # Vnm
            ifrag.Vnm.vp = ifrag.Vnm.vp_pot + ifrag.Vnm.vp_kin + ifrag.Vnm.vp_hxc
            self.Vnm.vp += ifrag.Vnm.vp
            self.Vnm.vp_pot += ifrag.Vnm.vp
            self.Vnm.vp_kin += ifrag.Vnm.vp
            self.Vnm.vp_hxc += ifrag.Vnm.vp
            self.Vnm.vp_xc += ifrag.Vnm.vp

            # Plotter
            if self.plot_things:
                ifrag.Plotter.vp = ifrag.Plotter.vp_pot + ifrag.Plotter.vp_kin + ifrag.Plotter.vp_hxc
                self.Plotter.vp     += ifrag.Plotter.vp     * ifrag.Plotter.Q
                self.Plotter.vp_pot += ifrag.Plotter.vp_pot * ifrag.Plotter.Q
                self.Plotter.vp_kin += ifrag.Plotter.vp_kin * ifrag.Plotter.Q
                self.Plotter.vp_hxc += ifrag.Plotter.vp_hxc * ifrag.Plotter.Q
                self.Plotter.vp_x   += ifrag.Plotter.vp_x   * ifrag.Plotter.Q
                self.Plotter.vp_c   += ifrag.Plotter.vp_c   * ifrag.Plotter.Q
                self.Plotter.vp_h   += ifrag.Plotter.vp_h   * ifrag.Plotter.Q

        vp = self.V.vp
    # elif self.optPart.vp_type == "potential_inversion":
    #     pass

    return  vp