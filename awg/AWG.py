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
        '_R',         # grating radius of carvature (focal length)
        '_d',         # array aperture spacing
        '_g',         # gap width between array apertures
        '_L0',        # minimum waveguide length offset (def. 0)
        '_Ni',        # number of input waveguides
        '_wi',        # input waveguide aperture width
        '_di',        # input waveguide spacing (def. 0)
        '_li',        # input waveguide offset spacing (def. 0)
        '_No',        # number of output waveguides
        '_wo',        # output waveguide aperture width
        '_do',        # output waveguide spacing (def. 0)
        '_lo',        # output waveguide offset spacing (def. 0)
        '_df',        # radial defocus (def. 0)
        '_confocal',  # use confocal arrangement rather than Rowland (def. false)
        '_wa',        # waveguide aperture width
        '_dl',        # waveguide length increment
        '_ns',        # slab index at center wavelength
        '_nc',        # core index at center wavelength
        '_Ng',        # core group index at center wavelength
        '_Ri',        # input/output radius curvature
        '_Ra'        # array radius curvature
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
            if (type(kwargs["N"]) == int) and (kwargs["N"] > 0):
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

        if "R" in _in:
            if ((type(kwargs["R"]) == int) or (type(kwargs["R"]) == float)) and (kwargs["R"] > 0):
                self._R = kwargs["R"]
            else:
                raise ValueError("The grating radius of curvature (focal length) 'R' [um] must be a positive float or integer")
        else:
            self._R = 100

        if "d" in _in:
            if ((type(kwargs["d"]) == int) or (type(kwargs["d"]) == float)) and (kwargs["d"] > 0):
                self._d = kwargs["d"]
            else:
                raise ValueError("The array aperture spacing 'd' [um] must be a positive float or integer")
        else:
            self._d = 1.3

        if "g" in _in:
            if ((type(kwargs["g"]) == int) or (type(kwargs["g"]) == float)) and (kwargs["g"] > 0):
                self._g = kwargs["g"]
            else:
                raise ValueError("The gap width between array aperture 'g' [um] must be a positive float or integer")
        else:
            self._g = 0.2

        if "L0" in _in:
            if ((type(kwargs["L0"]) == int) or (type(kwargs["L0"]) == float)) and (kwargs["L0"] >= 0):
                self._L0 = kwargs["L0"]
            else:
                raise ValueError("The minimum lenght offset 'L0' [um] must be a non-negative float or integer")
        else:
            self._L0 = 0

        if "Ni" in _in:
            if (type(kwargs["Ni"]) == int) and (kwargs["Ni"] >= 0):
                self._L0 = kwargs["Ni"]
            else:
                raise ValueError("The number of input waveguide 'Ni' must be a positive integer")
        else:
            self._Ni = 1

        if "wi" in _in:
            if ((type(kwargs["wi"]) == int) or (type(kwargs["wi"]) == float)) and (kwargs["wi"] > 0):
                self._wi = kwargs["wi"]
            else:
                raise ValueError("The input waveguide aperture width 'wi' [um] must be a positive float or integer")
        else:
            self._wi = 1

    @property
    def lambda_c(self):
        return self._lambda_c

    @lambda_c.setter
    def lambda_c(self,lambda_c):
        if (type(lambda_c) == float) or (type(lambda_c) == int) and (lambda_c > 0):
            self._lambda_c = lambda_c 
        else:
            raise ValueError("The central wavelength [um] must be a positive float or integer.")

    @property
    def clad(self):
        return self._clad
    
    @clad.setter
    def clad(self,clad):
        if (type(clad) == types.FunctionType) or (str(type(clad)) == "<class 'material.Material.Material'>") or (type(clad) == float) or (type(clad) == int):
            self._clad = clad
        else:
            raise ValueError("The cladding must be a material or a float representing its refractive index.")

    @property
    def core(self):
        return self._core
    
    @core.setter
    def core(self,core):
        if (type(core) == types.FunctionType) or (str(type(core)) == "<class 'material.Material.Material'>") or (type(core) == float) or (type(core) == int):
            self._core = core
        else:
            raise ValueError("The core must be a material or a float representing its refractive index.")

    @property
    def subs(self):
        return self._subs
    
    @clad.setter
    def subs(self,subs):
        if (type(subs) == types.FunctionType) or (str(type(subs)) == "<class 'material.Material.Material'>") or (type(subs) == float) or (type(subs) == int):
            self._subs = subs
        else:
            raise ValueError("The substrate must be a material or a float representing its refractive index.")

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self,w):
        if ((type(w) == int) or (type(w) == float)) and (w > 0):
            self._w = w 
        else:
            raise ValueError("The array waveguide core width 'w' [um] must be positive and be a float or an integer.")


    @property
    def h(self):
        return self._h

    @h.setter
    def h(self,h):
        if ((type(h) == int) or (type(h) == float)) and (h > 0):
            self._h = h 
        else:
            raise ValueError("The array waveguide core height 'h' [um] must be positive and be a float or an integer.")

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self,t):
        if ((type(t) == int) or (type(t) == float)) and (t >= 0):
            self._t = t 
        else:
            raise ValueError("The array waveguide slab thickness 't' (for rib waveguides) [um] must be non-negative and be a float or an integer.")

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self,N):
        if (type(N) == int) and (N > 0):
            self._N = N 
        else:
            raise ValueError("The The number of arrayed waveguide 'N' must be a positive integer.")

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self,m):
        if ((type(m) == int) or (type(m) == float)) and (m > 0):
            self._m = m 
        else:
            raise ValueError("The order of diffraction 'm' must be a positive integer.")
    
    @property
    def R(self):
        return self._R

    @R.setter
    def R(self,R):
        if ((type(R) == int) or (type(R) == float)) and (R > 0):
           self._R = R 
        else:
            raise ValueError("The grating radius of curvature (focal length) 'R' [um] must be a positive float or integer")

    @property
    def d(self):
        return self._d

    @d.setter
    def d(self,d):
        if ((type(d) == int) or (type(d) == float)) and (d > 0):
           self._d = d 
        else:
            raise ValueError("The array aperture spacing 'd' [um] must be a positive float or integer")

    @property
    def g(self):
        return self._g

    @g.setter
    def g(self,g):
        if ((type(g) == int) or (type(g) == float)) and (g > 0):
           self._g = g 
        else:
            raise ValueError("The gap width between array aperture 'g' [um] must be a positive float or integer")
    
    @property
    def L0(self):
        return self._L0

    @L0.setter
    def L0(self,L0):
        if ((type(L0) == float) or (type(L0) == int) and (L0 >=0)):
            self._L0 = L0
        else:
            raise ValueError("The minimum lenght offset 'L0' [um] must be a non-negative float or integer")
    
    @property
    def Ni(self):
        return self._Ni

    @Ni.setter
    def Ni(self,Ni):
        if (type(Ni) == int) and (Ni > 0):
            self._Ni = Ni 
        else:
            raise ValueError("The The number of input waveguide 'Ni' must be a positive integer.")
    
    @property
    def wi(self):
        return self._wi

    @wi.setter
    def wi(self,wi):
        if ((type(wi) == int) or (type(wi) == float)) and (wi > 0):
           self._wi = wi
        else:
            raise ValueError("The input waveguide aperture width 'wi' [um] must be a positive float or integer")

A = AWG(d = 1.6)
A.Ni = 1
print(A.m,A.d, A.L0)

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
