"""
energy.py
"""

def energy(self):
    """
    Gathers energies from all fragments
    """

    self.E.Etot = 0.0 
    self.E.Ekin = 0.0 
    self.E.Enuc = 0.0
    self.E.Evxc = 0.0
    self.E.Evha = 0.0

    for ifrag, j in enumerate(self.frags):

        Ekin = 0.0 
        Enuc = 0.0
        Evxc = 0.0
        Evha = 0.0
        Etot = 0.0

        # Go through the energy of each ensemble
        if not self.ens:
            frags = [ifrag]
        else:
            frags = [ifrag, self.efrags[j]]

        # Sum of fragment energies
        for i in frags:
            Ekin = i.E.Ekin * i.scale
            Enuc = i.E.Enuc * i.scale
            Evxc = i.E.Evxc * i.scale
            Evha = i.E.Evha * i.scale
            Etot = i.E.Etot * i.scale

        self.E.Etot += Etot
        self.E.Ekin += Ekin
        self.E.Evxc += Evxc
        self.E.Evha += Evha

    
    self.partition_energy()

        
        
