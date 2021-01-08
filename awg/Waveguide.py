from core import *
from material import *
import types

class Waveguide:
	__slots__ = [
				"_clad",
				"_core",
				"_subs",
				"_w",
				"_h",
				"_t"]
    def __init__(self,**kwargs):
        _in = kwargs.keys()

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
            self._w = 0.500

        if "h" in _in:
            if ((type(kwargs["h"]) == int) or (type(kwargs["h"]) == float)) and (kwargs["h"] > 0):
                self._h = kwargs["h"]
            else: 
                raise ValueError("The array waveguide core height 'h' [um] must be positive and be a float or an integer.")
        else:
            self._h = 0.200
        
        if "t" in _in:
            if ((type(kwargs["t"]) == int) or (type(kwargs["t"]) == float)) and (kwargs["t"] >= 0):
                self._t = kwargs["t"]
            else: 
                raise ValueError("The array waveguide slab thickness 't' (for rib waveguides) [um] must be non-negative and be a float or an integer.")
        else:
            self._t = 0

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
