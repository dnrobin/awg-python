from .core import *
from .material import *
from . import Field, Waveguide, Aperture
import types


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
        '_df',        # radial defocus
        '_ns',        # slab index at center wavelength
        '_nc',        # core index at center wavelength
        '_Ng',        # core group index at center wavelength
        '_Ri',        # input/output radius curvature
        '_Ra'         # array radius curvature
    ]
    def __init__(self,**kwargs):
        _in = kwargs.keys()

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

        if "nc" in _in:
            if ((type(kwargs["nc"]) == int) or (type(kwargs["nc"]) == float)) and (kwargs["nc"] > 0):
                self._nc = kwargs["nc"]
            else: 
                raise ValueError("The core index 'nc' must be positive and be a float or an integer.")
        else:
            if (type(self._core) == types.FunctionType) or (str(type(self._core)) == "<class 'material.Material.Material'>"):
                self._nc = self._core(self._lambda_c)
            else:
                self._nc = self._core

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
            self._h = 0.220
        
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
                self._Ni = kwargs["Ni"]
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

        if "di" in _in:
            if ((type(kwargs["di"]) == int) or (type(kwargs["di"]) == float)) and (kwargs["di"] > 0):
                self._di = kwargs["di"]
            else:
                raise ValueError("The input waveguide spacing 'di' [um] must be a positive float or integer")
        else:
            self._di = 0

        if "li" in _in:
            if ((type(kwargs["li"]) == int) or (type(kwargs["li"]) == float)) and (kwargs["li"] >= 0):
                self._li = kwargs["li"]
            else:
                raise ValueError("The input waveguide offset spacing 'li' [um] must be a non-negative float or integer")
        else:
            self._li = 0

        if "No" in _in:
            if (type(kwargs["No"]) == int) and (kwargs["No"] >= 0):
                self._No = kwargs["No"]
            else:
                raise ValueError("The number of output waveguide 'No' must be a positive integer")
        else:
            self._No = 1

        if "wo" in _in:
            if ((type(kwargs["wo"]) == int) or (type(kwargs["wo"]) == float)) and (kwargs["wo"] > 0):
                self._wo = kwargs["wo"]
            else:
                raise ValueError("The output waveguide aperture width 'wo' [um] must be a positive float or integer")
        else:
            self._wo = 1

        if "do" in _in:
            if ((type(kwargs["do"]) == int) or (type(kwargs["do"]) == float)) and (kwargs["do"] > 0):
                self._do = kwargs["do"]
            else:
                raise ValueError("The output waveguide spacing 'do' [um] must be a positive float or integer")
        else:
            self._do = 0

        if "lo" in _in:
            if ((type(kwargs["lo"]) == int) or (type(kwargs["lo"]) == float)) and (kwargs["lo"] >= 0):
                self._lo = kwargs["lo"]
            else:
                raise ValueError("The output waveguide offset spacing 'lo' [um] must be a non-negative float or integer")
        else:
            self._lo = 0

        if "confocal" in _in:
            if kwargs["confocal"] in [True, False]:
                self._confocal = kwargs["confocal"]
            else:
                raise ValueError("The confocal arrangement use instead of Rowland circle must be either True or False")
        else:
            self._confocal = False

        if "wa" in _in:
            if ((type(kwargs["wa"]) == int) or (type(kwargs["wa"]) == float)) and (kwargs["wa"] > 0):
                self._wa = kwargs["wa"]
            else:
                raise ValueError("The waveguide aperture width 'wa' [um] must be a positive float or integer")
        else:
            self._wa = 1 # Check what default value is


        if "dl" in _in:
            if ((type(kwargs["dl"]) == int) or (type(kwargs["dl"]) == float)) and (kwargs["dl"] > 0):
                self._dl = kwargs["dl"]
            else:
                raise ValueError("The arrayed waveguide lenght increment 'dl' must be a positive float or integer")
        else:
            self._dl = self._m*self._lambda_c/self._nc

        if "df" in _in:
            if ((type(kwargs["df"]) == int) or (type(kwargs["df"]) == float)) and (kwargs["df"] > 0):
                self._df = kwargs["df"]
            else:
                raise ValueError("The radial defocus 'df' must be a positive float or integer")
    # Other variable to do, must make the waveguide and arrayedwaveguide classes first
    def getSlabWaveguide(self):
        return Waveguide.Waveguide(clad = self._clad,core = self._core,subs = self._subs,h = self._h, t = self._t)
    
    def getArrayWaveguide(self):
        return Waveguide.Waveguide(clad = self._clad,core = self._core,subs = self._subs,w = self._w,h = self._h, t = self._t)

    def getInputAperture(self):
        return Aperture.Aperture(clad = self._clad,core = self._core,subs = self._subs,w = self._wi,h = self._h)


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

    @property
    def di(self):
        return self._di

    @di.setter
    def di(self,di):
        if ((type(di) == int) or (type(di) == float)) and (di > 0):
           self._di = di 
        else:
            raise ValueError("The input waveguide spacing 'di' [um] must be a positive float or integer")

    @property
    def li(self):
        return self._li

    @li.setter
    def li(self,li):
        if ((type(li) == int) or (type(li) == float)) and (li >= 0):
           self._li = li 
        else:
            raise ValueError("The input waveguide offset spacing 'li' [um] must be a non-negative float or integer")

    @property
    def No(self):
        return self._No

    @No.setter
    def No(self,Ni):
        if (type(No) == int) and (No > 0):
            self._No = No 
        else:
            raise ValueError("The The number of output waveguide 'No' must be a positive integer.")
    
    @property
    def wo(self):
        return self._wo

    @wo.setter
    def wo(self,wi):
        if ((type(wo) == int) or (type(wo) == float)) and (wo > 0):
           self._wo = wo
        else:
            raise ValueError("The output waveguide aperture width 'wo' [um] must be a positive float or integer")

    @property
    def do(self):
        return self._do

    @do.setter
    def do(self,do):
        if ((type(do) == int) or (type(do) == float)) and (do > 0):
           self._do = do 
        else:
            raise ValueError("The output waveguide spacing 'do' [um] must be a positive float or integer")

    @property
    def lo(self):
        return self._lo

    @lo.setter
    def lo(self,lo):
        if ((type(lo) == int) or (type(lo) == float)) and (lo >= 0):
           self._lo = lo 
        else:
            raise ValueError("The output waveguide offset spacing 'lo' [um] must be a non-negative float or integer")


    @property
    def confocal(self):
        return self._confocal

    @confocal.setter
    def confocal(self, confocal):
        if confocal in [True, False]:
            self._confocal = confocal
        else:
            raise ValueError("The confocal arrangement use instead of Rowland circle must be either True or False")

    @property
    def wa(self):
        return self._wa

    @wa.setter
    def wa(self,wa):
        if ((type(wa) == int) or (type(wa) == float)) and (wa > 0):
            self._wi = wa
        else:
            raise ValueError("The waveguide aperture width 'wa' [um] must be a positive float or integer")

    @property
    def dl(self):
        return self._dl

    @dl.setter
    def dl(self,dl):
        if ((type(dl) == int) or (type(dl) == float)) and (dl > 0):
            self._dl = dl
        else:
            raise ValueError("The arrayed waveguide lenght increment 'dl' must be a positive float or integer")

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self,df):
        if ((type(kwargs["df"]) == int) or (type(kwargs["df"]) == float)) and (kwargs["df"] > 0):
            self._df = kwargs["df"]
        else:
            raise ValueError("The radial defocus 'df' must be a positive float or integer")
    



def iw(model, lmbda, _input = 0, u = np.array([]),**kwargs):

    _in = kwargs.keys()


    if (type(_input) == int):
        if _input +1 > model.Ni:
            raise ValueError(f"Undefined input number {_input} for AWG having  {model.Ni} inputs.")

    offset = model.li + (_input-(model.Ni-1)/2)*max(model.di,model.wi)

    if str(type(u)) == "<class 'awg.Field.Field'>":
        F = u
        print(type(u))
    elif len(u) == 0:
        pass
    elif (min(u.shape) > 2) or (len(u.shape) > 2) :
        print((min(u.shape) > 2), (len(u.shape) > 2),u.shape)
        raise ValueError("Data provided for the input field must be a two column matrix of coordinate, value pairs.")
    else:
        n,m = u.shape
        F = Field.Field(u[:,0],u[:,1])

    if "ModeType" in _in:
        ModeType = kwargs["ModeType"]
    else:
        ModeType = "gaussian"

    if ModeType not in ["rect","gaussian", "solve"]:
        raise ValueError(f"Wrong mode type {ModeType}.")

    if "points" in _in:
        points = kwargs["points"]
    else:
        points = 100


    x = np.linspace(-1,1,points)*max(model.di,model.wi)
    F = model.getInputAperture().mode(x,np.zeros(len(x)))








def aw():
    pass

def ow():
    pass

def fpr1():
    pass

def fpr2():
    pass
