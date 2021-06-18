"""
vp_kinetic.py
"""

import numpy as np
import psi4

import sys

def vp_kinetic(self):
    """
    Calculates the kinetic component of vp
    """

    if self.optPart.kinetic_type == "vonweiz":
        raise ValueError("VonWeizacker not yet implemented")

    elif self.optPart.kinetic_type == "libxc_ke" or self.optPart.kinetic_type == 'param_ke':
        raise ValueError("Kinetic Energy functional not implemented")

    elif self.optPart.kinetic_type == "inversion":
        # if self.optPart.verbose and self.iteration == 1:
        #     print("\t\t\tCalculating the vp kinetic")

        # Individual vt
        for ifrag in self.frags:
            ifrag.V.vt =       - ( ifrag.V.vhxc + ifrag.V.vnuc)
            ifrag.Vnm.vt =     - ( ifrag.Vnm.vh + ifrag.Vnm.Vxca + ifrag.Vnm.V )

            if self.plot_things:
                ifrag.Plotter.vt = - ( ifrag.Plotter.vhxc + ifrag.Plotter.vnuc)

        # Set target components
        self.inverter.dt = [self.dfa, self.dfb]
        self.inverter.ct = []
        self.inverter.eigvecs_a = []
        self.inverter.eigvecs_b = []


        # veff_guess = vha + vc + vx #+ vn
        # veff_guess = self.grid.dft_grid_to_fock_one(veff_guess, self.grid.vpot)
        #veff_guess = np.zeros((self.nbf))
        # self.inverter.v0 = np.concatenate((veff_guess, veff_guess))

        # # Set up required matrices
        # self.inverter.T = np.zeros_like(self.dfa)
        # self.inverter.V = np.zeros_like(self.dfb)
        # for ifrag in self.frags:
        #     self.inverter.T += ifrag.V.Tnm
        #     self.inverter.V += ifrag.V.Vnm

        # Invert protomolecular density

        _, vha = self.grid.esp(Da=self.dfa, Db=self.dfb, grid=self.grid.plot_points)
        _, vha_dft = self.grid.esp(Da=self.dfa, Db=self.dfb, vpot=self.grid.vpot)
        vc = self.grid.vxc(1,  Da=self.dfa, Db=self.dfb, grid=self.grid.plot_points)
        vx = self.grid.vxc(12, Da=self.dfa, Db=self.dfb, grid=self.grid.plot_points)
        vn = self.Plotter.vnuc

        self.Plotter.vh0 = vha
        self.Plotter.vxc0 = vx + vc
        self.Plotter.vext0 = vn
    

        if self.optPart.inv_method == 'oucarter':
            print("\t\t\t\tStartiting Ou Carter inversion")


            vxc_dft, vxc_plotter = self.inverter.invert(method=self.optPart.inv_method, opt_method=self.optPart.opt_method)


            if self.ref == 2:
                print("Shifting")
                vxc_plotter[0][:] -= vxc_plotter[0][0]
                vxc_plotter[1][:] -= vxc_plotter[1][0]
                vxc_dft[0][:] -= vxc_dft[0][0]
                vxc_dft[1][:] -= vxc_dft[0][1]
                # vxc_plotter[0,:] -= vxc_plotter[0,0]
                # vxc_dft[0,:] -= vxc_dft[0,0]
                # vxc_plotter[1,:] -= vxc_plotter[1,0]
                # vxc_dft[1,:] -= vxc_dft[1,0]

                vxc_plotter  = vxc_plotter[0][:]
                vxc_dft      = vxc_dft[0][:]

            else:
                vxc_plotter -= vxc_plotter[0]
                vxc_dft -= vxc_dft[0]

            self.Plotter.vxc = vxc_plotter
            self.V.vxc       = vxc_dft

            #We displace potential to move it to zero, while we work on mu
            self.Plotter.vt = - (vxc_plotter   + vha     + vn)
            self.V.vt       = - (vxc_dft       + vha_dft + self.V.vnuc) 
            self.Vnm.vt     = - (self.inverter.Vxca_inv + self.inverter.va + self.Vnm.V )

            #Build KSmatrix of full system and all the energies.
            # Hartree
            # J = np.einsum('pqrs,rs->pq', self.I, self.dfa, optimize=True)
            # self.grid.vpot.set_D([psi4.core.Matrix.from_array(self.dfa)])
            # vxc = psi4.core.Matrix( self.nbf, self.nbf )
            # self.grid.vpot.compute_V([ vxc ])


            # veff = self.Tnm.np + self.Vnm.np + J + vxc
            # _, _ ,proto_density, _ = self.diagonalize( veff, self.nalpha )
            # self.wft_density = proto_density


        elif self.optPart.inv_method == 'wuyang':
            self.inverter.invert(initial_guess='fermi_amaldi', method=self.optPart.inv_method, opt_method=self.optPart.opt_method)
            vt_plotter = self.grid.on_grid_ao(self.inverter.v, grid=self.grid.plot_points)
            vt         = self.grid.on_grid_ao(self.inverter.v, vpot=self.grid.vpot)
            self.V.vt = vt + vn
            self.Plotter.vt = vt_plotter + self.Plotter.vnuc


    # Finalize vp_kin
    for ifrag in self.frags:
        ifrag.V.vp_kin       = self.V.vt       - ifrag.V.vt
        ifrag.Vnm.vp_kin     = self.Vnm.vt - ifrag.Vnm.vt

        if self.plot_things:
            ifrag.Plotter.vp_kin = self.Plotter.vt - ifrag.Plotter.vt

