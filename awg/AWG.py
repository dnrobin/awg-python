from .core import *

class AWG:
    __slots__ = [
        'lambda_c'  # center wavelength
        'w'         # waveguide core width
        'h'         # waveguide core height
        't'         # waveguide slab thickness (for rib waveguides) (def. 0)
        'N'         # number of arrayed waveguides
        'm'         # diffraction order
        'R'         # grating radius of carvature (focal length)
        'd'         # array aperture spacing
        'g'         # gap width between array apertures
        'L0'        # minimum waveguide length offset (def. 0)
        'Ni'        # number of input waveguides
        'wi'        # input waveguide aperture width
        'di'        # input waveguide spacing (def. 0)
        'li'        # input waveguide offset spacing (def. 0)
        'No'        # number of output waveguides
        'wo'        # output waveguide aperture width
        'do'        # output waveguide spacing (def. 0)
        'lo'        # output waveguide offset spacing (def. 0)
        'df'        # radial defocus (def. 0)
        'confocal'  # use confocal arrangement rather than Rowland (def. false)
        'wa'        # waveguide aperture width
        'dl'        # waveguide length increment
        'ns'        # slab index at center wavelength
        'nc'        # core index at center wavelength
        'Ng'        # core group index at center wavelength
        'Ri'        # input/output radius curvature
        'Ra'        # array radius curvature
    ]
    def __init__(self):
        pass

def iw():
    pass

def aw():
    pass

def ow():
    pass

def fpr1():
    pass

def fpr2():
    pass
