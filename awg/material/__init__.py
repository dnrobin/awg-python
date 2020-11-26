"""
awg.material is a package for modeling material chromatic dispersion.
"""

from .Material import Material

# here we define some pre-existing material functions

def Air(wvl):
    pass

def Si(wvl):
    pass

def SiO2(wvl):
    pass

def Si3N4(wvl):
    pass

def Ge(wvl):
    pass

def dispersion(function, wvl1, wvl2):
    """calculates chromatic dispersion curve over wavelength range.
    """
    pass