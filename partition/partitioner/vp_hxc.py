"""
vp_hxc.py
"""

import numpy as np
import psi4

def vp_hxc(self):
    """
    Calculates hxc component of vp
    """

    # Calculate hxc functional for promolecular density on grid

    #Hartree Exchange Correlation on DFT GRID
    _, self.V.vh = self.grid.esp(Da=self.inverter.dt[0], Db=self.inverter.dt[1], vpot=self.grid.vpot)
    self.V.vx    = self.grid.vxc(func_id=1 , Da=self.inverter.dt[0], Db=self.inverter.dt[1], vpot=self.grid.vpot).T
    self.V.vc    = self.grid.vxc(func_id=12, Da=self.inverter.dt[0], Db=self.inverter.dt[1], vpot=self.grid.vpot).T   

    # wfn_dummy = psi4.energy('svwn/'+self.basis_str, molecule=self.mol, return_wfn=True)[1]

    # if self.ref == 1:
    #     df_psi4 = psi4.core.Matrix.from_array([ self.df ])
    #     wfn_dummy.V_potential().set_D( [ df_psi4 ] )
    #     vxc_a = psi4.core.Matrix( self.nbf, self.nbf )
    #     self.grid.wfn.V_potential().compute_V([vxc_a])
    #     self.Vnm.Vxca = vxc_a
    #     self.Vnm.Vxcb = vxc_a
    # else:
    dfa_psi4 = psi4.core.Matrix.from_array([ self.inverter.dt[0] ])
    dfb_psi4 = psi4.core.Matrix.from_array([ self.inverter.dt[1] ])
    self.grid.wfn.V_potential().set_D([dfa_psi4, dfb_psi4])
    vxc_a = psi4.core.Matrix( self.nbf, self.nbf )
    vxc_b = psi4.core.Matrix( self.nbf, self.nbf )
    self.grid.wfn.V_potential().compute_V([vxc_a, vxc_b])
    self.frags_exc = self.grid.wfn.V_potential().quadrature_values()['FUNCTIONAL']
    self.Vnm.Vxca = vxc_a
    self.Vnm.Vxcb = vxc_b

    self.Vnm.vh = self.inverter.va
    # Need vxc of fragment densities

    #Hartree Exchange Correlation on Plotting GRID
    if self.plot_things:
        _, self.Plotter.vh = self.grid.esp(Da=self.inverter.dt[0], Db=self.inverter.dt[1], grid=self.grid.plot_points)
        self.Plotter.vx    = self.grid.vxc(func_id=1 , Da=self.inverter.dt[0], Db=self.inverter.dt[1], grid=self.grid.plot_points).T
        self.Plotter.vc    = self.grid.vxc(func_id=12, Da=self.inverter.dt[0], Db=self.inverter.dt[1], grid=self.grid.plot_points).T   

    for ifrag in self.frags:
        ifrag.V.vp_h = self.V.vh - ifrag.V.vh
        ifrag.V.vp_x = self.V.vx - ifrag.V.vx
        ifrag.V.vp_c = self.V.vc - ifrag.V.vc
        ifrag.V.vp_hxc = ifrag.V.vp_h + ifrag.V.vp_x + ifrag.V.vp_c

        # Vnm
        ifrag.Vnm.vp_h = self.Vnm.vh - ifrag.Vnm.vh
        ifrag.Vnm.vp_xc = self.Vnm.Vxca - ifrag.Vnm.Vxca
        ifrag.Vnm.vp_hxc = ifrag.Vnm.vp_h + ifrag.Vnm.vp_xc

        if self.plot_things:
            ifrag.Plotter.vp_h = self.Plotter.vh - ifrag.Plotter.vh
            ifrag.Plotter.vp_x = self.Plotter.vx - ifrag.Plotter.vx
            ifrag.Plotter.vp_c = self.Plotter.vc - ifrag.Plotter.vc
            ifrag.Plotter.vp_hxc = ifrag.Plotter.vp_h + ifrag.Plotter.vp_x + ifrag.Plotter.vp_c

