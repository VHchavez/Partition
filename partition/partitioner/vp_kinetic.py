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

        # Fragmetns vt
        for ifrag in self.frags:
            ifrag.V.vt =       - ( ifrag.V.vhxc + ifrag.V.vnuc)
            ifrag.Vnm.vt =     - ( ifrag.Vnm.vh + ifrag.Vnm.Vxca + ifrag.Vnm.V )

            if self.plot_things:
                ifrag.Plotter.vt = - ( ifrag.Plotter.vhxc + ifrag.Plotter.vnuc)

        # Set target components
        # target_a = self.molSCF.da
        # target_b = self.molSCF.db
        _, vha = self.grid.esp(Da=self.inverter.dt[0], Db=self.inverter.dt[1], grid=self.grid.plot_points)
        _, vha_dft = self.grid.esp(Da=self.inverter.dt[0], Db=self.inverter.dt[1], vpot=self.grid.vpot)
        vc = self.grid.vxc(1,  Da=self.inverter.dt[0], Db=self.inverter.dt[1], grid=self.grid.plot_points)
        vx = self.grid.vxc(12, Da=self.inverter.dt[0], Db=self.inverter.dt[1], grid=self.grid.plot_points)
        vn = self.Plotter.vnuc

        self.Plotter.vh0 = vha
        self.Plotter.vxc0 = vx + vc
        self.Plotter.vext0 = vn

        # Invert protomolecular density
        if self.optPart.inv_method == 'oucarter':
            print("\t\t\t\tStartiting Ou Carter inversion")

            vxc_dft, vxc_plotter = self.inverter.invert(initial_guess='hartree',method=self.optPart.inv_method, opt_method=self.optPart.opt_method)
            density_accuracy = np.linalg.norm(self.inverter.dt[0] -self.inverter.Da)
            print("\t\t\t\tInverted Potential matched density:", density_accuracy)

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
            self.inverter.invert(initial_guess=self.initial_guess, method=self.optPart.inv_method, opt_method=self.optPart.opt_method)
            density_accuracy = np.linalg.norm(self.inverter.dt[0] -self.inverter.Da)
            print("\t\t\t\tInverted Potential matched density:", density_accuracy)


            N = self.inverter.nalpha + self.inverter.nbeta

            # PLOT
            v_rest  = self.inverter.v_pbs
            v_guess_plot = (1-1/N) * vha
            v_rest_plot = self.grid.on_grid_ao( v_rest, grid=self.grid.plot_points )
            vxc_plot = v_guess_plot + v_rest_plot - vha
            hvxc_plot = v_guess_plot + v_rest_plot
            self.Plotter.rest = v_rest
            self.Plotter.inv_vxc = vxc_plot
            self.Plotter.inv_hvxc = hvxc_plot
            self.Plotter.vt = -(hvxc_plot + self.Plotter.vnuc)
            
            # DFT
            v_rest_dft = self.grid.on_grid_ao(v_rest, vpot=self.grid.vpot)
            v_guess_dft = (1-1/N) * vha_dft
            hvxc_dft = v_rest_dft + v_guess_dft
            self.V.vt = -(hvxc_dft + self.V.vnuc)

            # NM
            self.Vnm.vt = -(self.inverter.inv_vksa)

        elif self.optPart.inv_method == 'wuyang_pdft':
            print("I am inverting with wuyang pdft")
            self.inverter.invert(initial_guess=self.initial_guess, method=self.optPart.inv_method, opt_method=self.optPart.opt_method)



    # Finalize vp_kin
    for ifrag in self.frags:
        ifrag.V.vp_kin       = self.V.vt   - ifrag.V.vt
        ifrag.Vnm.vp_kin     = self.Vnm.vt - ifrag.Vnm.vt

        if self.plot_things:
            ifrag.Plotter.vp_kin = self.Plotter.vt - ifrag.Plotter.vt

