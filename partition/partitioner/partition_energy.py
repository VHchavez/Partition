"""
partition_energy.py
"""

import numpy as np

def partition_energy(self):
    """
    Calculates the partition energy of the system
    """


    self.E.Ef_nuc = self.molE.Enuc # Sum of nuclear fragments is just nuclear of molecule
    self.E.Ef_ext = 0.0
    self.E.Ef_kin = 0.0
    self.E.Ef_vxc = 0.0
    self.E.Ef_vha = 0.0
    self.E.Ef_ext = 0.0

    self.E.Ep_ext = 0.0
    #devwarn Will only work for diatomics for now
    # External nuclear potential
    # self.E.Ep_ext += np.sum( (self.frags[0].da_frac + self.frags[0].db_frac) * self.frags[1].V.Vnm )
    # self.E.Ep_ext += np.sum( (self.frags[1].da_frac + self.frags[1].db_frac) * self.frags[0].V.Vnm )

    # Kinetic
    for ifrag in self.frags:
        self.E.Ef_ext += ifrag.E.Eext # External nuclear potential
        self.E.Ef_kin += ifrag.E.Ekin # Kinetic
        self.E.Ef_vxc += ifrag.E.Evxc # XC
        self.E.Ef_vha += ifrag.E.Evha # Hartree

    # Components of Ep
    self.E.Ep_kin = self.molE.Ekin - self.E.Ef_kin
    self.E.Ep_vxc = self.molE.Evxc - self.E.Ef_vxc
    self.E.Ep_vha = self.molE.Evha - self.E.Ef_vha
    self.E.Ep_ext = self.molE.Eext - self.E.Ef_ext

    # Ep and Ef
    self.E.Ep = self.E.Ep_kin + self.E.Ep_vha + self.E.Ep_vxc + self.E.Ep_ext
    self.E.Ef = self.frags[0].E.Etot + self.frags[1].E.Etot













    