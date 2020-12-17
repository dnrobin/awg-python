from core import *
from material import *
import types

class Test:
    def __init__(self):
        pass

class AWG:
    __slots__ = [
        'lambda_c',  # center wavelength
        'clad',      # clad material or index of the waveguide
        'core',      # core material or index of the waveguide
        'subs',      # substrate material or index of the waveguide
        'w',         # waveguide core width
        'h',         # waveguide core height
        't',         # waveguide slab thickness (for rib waveguides) (def. 0)
        'N',         # number of arrayed waveguides
        'm',         # diffraction order
        'R',         # grating radius of carvature (focal length)
        'd',         # array aperture spacing
        'g',         # gap width between array apertures
        'L0',        # minimum waveguide length offset (def. 0)
        'Ni',        # number of input waveguides
        'wi',        # input waveguide aperture width
        'di',        # input waveguide spacing (def. 0)
        'li',        # input waveguide offset spacing (def. 0)
        'No',        # number of output waveguides
        'wo',        # output waveguide aperture width
        'do',        # output waveguide spacing (def. 0)
        'lo',        # output waveguide offset spacing (def. 0)
        'df',        # radial defocus (def. 0)
        'confocal',  # use confocal arrangement rather than Rowland (def. false)
        'wa',        # waveguide aperture width
        'dl',        # waveguide length increment
        'ns',        # slab index at center wavelength
        'nc',        # core index at center wavelength
        'Ng',        # core group index at center wavelength
        'Ri',        # input/output radius curvature
        'Ra'        # array radius curvature
    ]
    def __init__(self,**kwargs):
        _in = kwargs.keys()
        print(_in)

        if "lambda_c" in _in:
            if (type(kwargs["lambda_c"]) == float) or (type(kwargs["lambda_c"]) == int):
                self.lambda_c = kwargs["lambda_c"]
            else:
                raise ValueError("The central wavelength [um] must be a float or an integer.")
        else:
            self.lambda_c = 1.550

        if "clad" in _in:
            if (type(kwargs["clad"]) == types.FunctionType) or (str(type(kwargs["clad"])) == "<class 'material.Material.Material'>") or (type(kwargs["clad"]) == float) or (type(kwargs["clad"]) == int):
                self.clad = kwargs["clad"]
            else:
                raise ValueError("The cladding must be a material or a float representing its refractive index.")
        else:
            self.clad = SiO2

        if "core" in _in:
            if (type(kwargs["core"]) == types.FunctionType) or (str(type(kwargs["core"])) == "<class 'material.Material.Material'>") or (type(kwargs["core"]) == float) or (type(kwargs["core"]) == int):
                self.core = kwargs["core"]
            else:
                raise ValueError("The core must be a material or a float representing its refractive index.")
        else:
            self.core = Si

        if "subs" in _in:
            if (type(kwargs["subs"]) == types.FunctionType) or (str(type(kwargs["subs"])) == "<class 'material.Material.Material'>") or (type(kwargs["subs"]) == float) or (type(kwargs["subs"]) == int):
                self.clad = kwargs["clad"]
            else:
                raise ValueError("The substrate must be a material or a float representing its refractive index.")
        else:
            self.subs = SiO2


A = AWG(lambda_c = 1.6)
print(A.clad)
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
