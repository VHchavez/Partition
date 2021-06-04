"""
vp_hxc.py
"""

import numpy as np

def vp_hxc(self):
    """
    Calculates hxc component of vp
    """

    # Calculate hxc functional for promolecular density on grid

    #Hartree Exchange Correlation on DFT GRID
    _, self.V.vh = self.grid.esp(Da=self.dfa, Db=self.dfb, vpot=self.grid.vpot)
    self.V.vx    = self.grid.vxc(func_id=1 , Da=self.dfa, Db=self.dfb, vpot=self.grid.vpot).T
    self.V.vc    = self.grid.vxc(func_id=12, Da=self.dfa, Db=self.dfb, vpot=self.grid.vpot).T   

    #Hartree Exchange Correlation on Plotting GRID
    _, self.Plotter.vh = self.grid.esp(Da=self.dfa, Db=self.dfb, grid=self.grid.plot_points)
    self.Plotter.vx    = self.grid.vxc(func_id=1 , Da=self.dfa, Db=self.dfb, grid=self.grid.plot_points).T
    self.Plotter.vc    = self.grid.vxc(func_id=12, Da=self.dfa, Db=self.dfb, grid=self.grid.plot_points).T   


    for ifrag in self.frags:
        ifrag.V.vp_h = self.V.vh - ifrag.V.vh
        ifrag.V.vp_x = self.V.vx - ifrag.V.vx
        ifrag.V.vp_c = self.V.vc - ifrag.V.vc
        ifrag.V.vp_hxc = ifrag.V.vp_h + ifrag.V.vp_x + ifrag.V.vp_c

        ifrag.Plotter.vp_h = self.Plotter.vh - ifrag.Plotter.vh
        ifrag.Plotter.vp_x = self.Plotter.vx - ifrag.Plotter.vx
        ifrag.Plotter.vp_c = self.Plotter.vc - ifrag.Plotter.vc
        ifrag.Plotter.vp_hxc = ifrag.Plotter.vp_h + ifrag.Plotter.vp_x + ifrag.Plotter.vp_c

