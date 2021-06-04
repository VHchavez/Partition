"""
ep_kinetic.py
"""

def ep_kinetic(self):
    """
    Calculates ep_kinetic per fragment
    """

    if self.kinetic_type == "vonweiz":
        raise ValueError("Von Weizacker not implemented yet")

    elif self.kinetic_type == "libxc_ke":
        raise ValueError("Kinetic Energy Functional from Libxc not implemented yet")

    elif self.kientic_type == "inversion":

        # # Calculate kinetic energy desntiy using laplacian of orbitals
        # self.ked = 0.0

        # #figure out number of occupied orbitals
        # noc = self.nalpha + self.nbeta

        # # Get kinetic energy density per fragment


        # # Moleculer kinetic Energy
        # tsm 

        self.E.Ep.kin = 0.0