from core import *
from material import *
import types

class Test:
    def __init__(self):
        pass

class AWG:
    __slots__ = [
        '_lambda_c',  # center wavelength
        '_clad',      # clad material or index of the waveguide
        '_core',      # core material or index of the waveguide
        '_subs',      # substrate material or index of the waveguide
        '_w',         # waveguide core width
        '_h',         # waveguide core height
        '_t',         # waveguide slab thickness (for rib waveguides) (def. 0)
        '_N',         # number of arrayed waveguides
        '_m',         # diffraction order
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
            if ((type(kwargs["lambda_c"]) == float) or (type(kwargs["lambda_c"]) == int)) and (kwargs["lambda_c"] > 0):
                self._lambda_c = kwargs["lambda_c"]
            else:
                raise ValueError("The central wavelength [um] must be a positive float or integer.")
        else:
            self._lambda_c = 1.550

        if "clad" in _in:
            if (type(kwargs["clad"]) == types.FunctionType) or (str(type(kwargs["clad"])) == "<class 'material.Material.Material'>") or (type(kwargs["clad"]) == float) or (type(kwargs["clad"]) == int):
                self._clad = kwargs["clad"]
            else:
                raise ValueError("The cladding must be a material or a float representing its refractive index.")
        else:
            self._clad = SiO2

        if "core" in _in:
            if (type(kwargs["core"]) == types.FunctionType) or (str(type(kwargs["core"])) == "<class 'material.Material.Material'>") or (type(kwargs["core"]) == float) or (type(kwargs["core"]) == int):
                self._core = kwargs["core"]
            else:
                raise ValueError("The core must be a material or a float representing its refractive index.")
        else:
            self._core = Si

        if "subs" in _in:
            if (type(kwargs["subs"]) == types.FunctionType) or (str(type(kwargs["subs"])) == "<class 'material.Material.Material'>") or (type(kwargs["subs"]) == float) or (type(kwargs["subs"]) == int):
                self._subs = kwargs["subs"]
            else:
                raise ValueError("The substrate must be a material or a float representing its refractive index.")
        else:
            self._subs = SiO2

        if "w" in _in:
            if ((type(kwargs["w"]) == int) or (type(kwargs["w"]) == float)) and (kwargs["w"] > 0):
                self._w = kwargs["w"]
            else: 
                raise ValueError("The array waveguide core width 'w' [um] must be positive and be a float or an integer.")
        else:
            self._w = 0.450

        if "h" in _in:
            if ((type(kwargs["h"]) == int) or (type(kwargs["h"]) == float)) and (kwargs["h"] > 0):
                self._h = kwargs["h"]
            else: 
                raise ValueError("The array waveguide core height 'h' [um] must be positive and be a float or an integer.")
        else:
            self._h = 0.450
        
        if "t" in _in:
            if ((type(kwargs["t"]) == int) or (type(kwargs["t"]) == float)) and (kwargs["t"] >= 0):
                self._t = kwargs["t"]
            else: 
                raise ValueError("The array waveguide slab thickness 't' (for rib waveguides) [um] must be non-negative and be a float or an integer.")
        else:
            self._t = 0

        if "N" in _in:
            if ((type(kwargs["N"]) == int) or (type(kwargs["N"]) == float)) and (kwargs["N"] > 0):
                self._N = kwargs["N"]
            else: 
                raise ValueError("The The number of arrayed waveguide 'N' must be a positive integer.")
        else:
            self._N = 40

        if "m" in _in:
            if ((type(kwargs["m"]) == int) or (type(kwargs["m"]) == float)) and (kwargs["m"] > 0):
                self._m = kwargs["m"]
            else: 
                raise ValueError("The order of diffraction 'm' must be a positive integer.")
        else:
            self._m = 30


    @property
    def lambda_c(self):
        return self._lambda_c

    @lambda_c.setter
    def lambda_c(self,lambda_c):
        if not((type(lambda_c) == float) or (type(lambda_c) == int) and (lambda_c > 0)):
            raise ValueError("The central wavelength [um] must be a positive float or integer.")
        else:
            self._lambda_c = lambda_c 

    @property
    def clad(self):
        return self._clad
    
    @clad.setter
    def clad(self,clad):
        if not ((type(clad) == types.FunctionType) or (str(type(clad)) == "<class 'material.Material.Material'>") or (type(clad) == float) or (type(clad) == int)):
            raise ValueError("The cladding must be a material or a float representing its refractive index.")
        else:
            self._clad = clad

    @property
    def core(self):
        return self._core
    
    @core.setter
    def clad(self,core):
        if not ((type(core) == types.FunctionType) or (str(type(core)) == "<class 'material.Material.Material'>") or (type(core) == float) or (type(core) == int)):
            raise ValueError("The core must be a material or a float representing its refractive index.")
        else:
            self._core = core

    @property
    def subs(self):
        return self._subs
    
    @clad.setter
    def clad(self,subs):
        if not ((type(subs) == types.FunctionType) or (str(type(subs)) == "<class 'material.Material.Material'>") or (type(subs) == float) or (type(subs) == int)):
            raise ValueError("The substrate must be a material or a float representing its refractive index.")
        else:
            self._subs = subs

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self,m):
        if m <= 0:
            raise ValueError("The order of diffraction 'm' must be a positive integer.")
        else:
            self._m = m 
    

A = AWG(lambda_c = 1.6, m = 5)
A.clad = Si
print(A.m,A.lambda_c, A.subs)
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
