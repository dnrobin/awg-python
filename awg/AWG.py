from .core import *
from .material import *
from .material.Material import Material
from . import Field, Waveguide, Aperture
import types
from numpy.random import randn

class AWG:
    __slots__ = [
        '_lambda_c',  # center wavelength
        '_clad',      # clad material or index of the waveguide
        'nclad',
        '_core',      # core material or index of the waveguide
        'ncore',
        '_subs',      # substrate material or index of the waveguide
        'nsubs',
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
        '_defocus',
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
            if (type(kwargs["clad"]) == types.FunctionType) or (type(kwargs["clad"]) == float) or (type(kwargs["clad"]) == int):
                self._clad = Material(kwargs["clad"])
            elif (str(type(kwargs["clad"])) == "<class 'awg.material.Material.Material'>"):
                self._clad = kwargs["clad"]
            elif type(kwargs["clad"]) == list:
                self._clad = Material(list_to_array(kwargs["clad"]))
            else:
                raise ValueError("The cladding must be a material or a float representing its refractive index.")
        else:
            self._clad = Material(SiO2)
        self.nclad = self.clad.index(self.lambda_c)

        if "core" in _in:
            if (type(kwargs["core"]) == types.FunctionType) or (type(kwargs["core"]) == float) or (type(kwargs["core"]) == int):
                self._core = Material(kwargs["core"])
            elif (str(type(kwargs["core"])) == "<class 'awg.material.Material.Material'>"):
                self._core = kwargs["core"]
            elif type(kwargs["core"]) == list:
                self._core = Material(list_to_array(kwargs["core"]))
            else:
                raise ValueError("The core must be a material or a float representing its refractive index.")
        else:
            self._core = Material(Si)
        self.ncore = self.core.index(self.lambda_c)

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
            if (type(kwargs["subs"]) == types.FunctionType) or (type(kwargs["subs"]) == float) or (type(kwargs["subs"]) == int):
                self._subs = Material(kwargs["subs"])
            elif (str(type(kwargs["subs"])) == "<class 'awg.material.Material.Material'>"):
                self._subs = kwargs["subs"]
            elif type(kwargs["subs"]) == list:
                self._subs = Material(list_to_array(kwargs["subs"]))
            else:
                raise ValueError("The substrate must be a material or a float representing its refractive index.")
        else:
            self._subs = Material(SiO2)
        self.nsubs = self.subs.index(self.lambda_c)

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

        if "defocus" in _in:
            self._defocus = kwargs["defocus"]
        else:
            self._defocus = 0

        if "wa" in _in:
            if ((type(kwargs["wa"]) == int) or (type(kwargs["wa"]) == float)) and (kwargs["wa"] > 0):
                self._wa = kwargs["wa"]
            else:
                raise ValueError("The waveguide aperture width 'wa' [um] must be a positive float or integer")
        else:
            self._wa = self._d - self._g

        wg = self.getArrayWaveguide()
        self._nc = wg.index(self.lambda_c,1)[0]

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
        return Waveguide.Waveguide(clad = self._clad,core = self._core,subs = self._subs,h = self._h, t = self._h)
    
    def getArrayWaveguide(self):
        return Waveguide.Waveguide(clad = self._clad,core = self._core,subs = self._subs,w = self._w,h = self._h, t = self._t)

    def getInputAperture(self):
        return Aperture.Aperture(clad = self._clad,core = self._core,subs = self._subs,w = self._wi,h = self._h)


    def getArrayAperture(self):
        return Aperture.Aperture(clad = self._clad,core = self._core,subs = self._subs,w = self._wa,h = self._h)

    def getOutputAperture(self):
        return Aperture.Aperture(clad = self._clad,core = self._core,subs = self._subs,w = self._wo,h = self._h)

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
    def nc(self):
        return self._nc
    

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
    def No(self,No):
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
    def defocus(self):
        return self._defocus
    
    @defocus.setter
    def defocus(self,defocus):
        if ((type(defocus) == int) or (type(defocus) == float)) and (defocus > 0):
            self._defocus = defocus
        else:
            raise ValueError("The defocus or R must be a positive float or integer")



    @property
    def wa(self):
        return self._wa

    @wa.setter
    def wa(self,wa):
        if ((type(wa) == int) or (type(wa) == float)) and (wa > 0):
            self._wa = wa
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

    if str(type(u)) == "<class 'awg.Field.Field'>":
        F = u
    elif len(u) == 0:
        pass
    elif (min(u.shape) > 2) or (len(u.shape) > 2) :
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
    F = model.getInputAperture().mode(lmbda, x= x, ModeType = ModeType)

    return F.normalize()

def fpr1(model,lmbda,F0,**kwargs):
    _in =kwargs.keys()

    if "x" in _in:
        x = kwargs["x"]
    else:
        x = []

    if "Input" in _in:
        _input = kwargs["Input"]
    else:
        _input = 0

    if "points" in _in:
        points = kwargs["points"]
    else:
        points = 250

    xi = F0.x
    ui = F0.Ex
    ns = model.getSlabWaveguide().index(lmbda,1)[0]

    if len(x) == 0:
        sf = np.linspace(-1/2,1/2,points)*(model.N+4)*model.d
    else:
        sf = x

    R = model.R
    r = model.R/2
    if model.confocal:
        r = model.R

    a = xi/r
    xp = r*np.tan(a)
    dp = r*(1/np.cos(a))-r
    up = ui*np.exp(1j*2*np.pi/lmbda*ns*dp)

    a = sf/R
    xf = R*np.sin(a)
    zf = model.defocus + R*np.cos(a)

    uf = diffract2(lmbda/ns,up,xp,xf,zf)[0]
    print(uf)




    """s0 = model.li + (_input-(model.Ni-1)/2)*max(model.di,model.wi)
    t0 = s0/r
    x0 = r*np.sin(t0)
    z0 = r*(1-np.cos(t0))
    #print(s0,t0,x0,z0,R,r,sep = "\n")
    t = sf/R
    x = R*np.sin(t)
    z = R*np.cos(t)
    #print(t,x,z,sep = "\n")
    a0 = np.arctan(np.sin(t0)/(1+np.cos(t0)))
    xf = (x+x0)*np.cos(a0)+(z+z0)*np.sin(a0)
    zf = -(x+x0)*np.sin(a0)+(z+z0)*np.cos(a0)

    uf = diffract(lmbda/ns,ui,xi,xf,zf)"""

    return Field.Field(sf,uf).normalize(F0.power())

def aw(model,lmbda,F0,**kwargs): # F0 = initial Field
    _in = kwargs.keys()

    if "ModeType" in _in:
        ModeType = kwargs["ModeType"]
    else:
        ModeType = "gaussian"

    if ModeType.lower() not in ["rect","gaussian", "solve"]:
        raise ValueError(f"Wrong mode type {ModeType}.")

    if "PhaseErrorVar" in _in: 
        PhaseErrorVar = kwargs["PhaseErrorVar"]
    else:
        PhaseErrorVar = 0

    if "InsertionLoss" in _in:
        InsertionLoss = kwargs["InsertionLoss"] # Insertion Loss in dB
    else:
        InsertionLoss = 0

    if "PropagationLoss" in _in:
        PropagationLoss = kwargs["PropagationLoss"]
    else:
        PropagationLoss = 0

    x0 = F0.x
    u0 = F0.Ex
    P0 = F0.power()
    #print(u0)
    k0 = 2*np.pi/lmbda
    nc = model.getArrayWaveguide().index(lmbda,1)[0]

    dr = model.R * (1/np.cos(x0/model.R)-1)
    dp0 = 2*k0*nc*dr
    u0 = u0*np.exp(-1j*dp0)

    pnoise = randn(1,model.N)[0]*np.sqrt(PhaseErrorVar)
    iloss = 10**(-abs(InsertionLoss)/10)
    
    Aperture = model.getArrayAperture()

    Ex = np.zeros(len(F0.E))

    for i in range(model.N):
        xc = (i - (model.N-1)/2)*model.d

        Fk =  Aperture.mode(lmbda,x = x0-xc, ModeType = ModeType)#.normalize()

        Ek = Fk.Ex *rect((x0-xc)/model.d)

        Ek = pnorm(Fk.x,Ek)

        t = overlap(x0,u0,Ek)

        L = i*model.dl + model.L0
        phase = k0*nc*L+pnoise[i]

        ploss = 10**(-abs(PropagationLoss*L*1e-4)/10)
        t = t*ploss*iloss**2



        Efield = P0*t*Ek*np.exp(-1j*phase)

        Ex = Ex + Efield

    return Field.Field(x0,Ex)


def fpr2(model,lmbda,F0,**kwargs):
    _in  = kwargs.keys()
    
    if "x" in _in:
        x = kwargs["x"]
    else:
        x = []

    if "points" in _in:
        points = kwargs["points"]
    else:
        points = 250

    xi = F0.x
    ui = F0.Ex

    ns = model.getSlabWaveguide().index(lmbda,1)[0]
    nc = model.getArrayWaveguide().index(lmbda,1)[0]

    R = model.R
    r = R/2
    if model.confocal:
        r = R

    if len(x) == 0:
        sf = np.linspace(-1/2,1/2,points)*(model.No+4)*max(model.do,model.wo)
    else:
        sf = x

    uf = 0

    xf = r*np.sin(sf/r)
    zf = r*(1+np.cos(sf/r))
    uf = diffract(lmbda/ns,ui,xi,xf,zf)

    return Field.Field(sf,uf).normalize(F0.power())


def ow(model,lmbda,F0,**kwargs):

    if "ModeType" in kwargs.keys():
        ModeType = kwargs["ModeType"]
    else:
        ModeType = "gaussian"

    if ModeType.lower() not in ["rect","gaussian", "solve"]:
        raise ValueError(f"Wrong mode type {ModeType}.")

    x0 = F0.x
    u0 = F0.Ex
    P0 = F0.power()

    Aperture = model.getOutputAperture()

    T = np.zeros(model.No)

    for i in range(model.No):

        xc = model.lo +(i-(model.No-1)/2)*max(model.do,model.wo)

        Fk = Aperture.mode(lmbda,x = x0-xc, ModeType = ModeType)
        Ek = Fk.Ex

        Ek = Ek*rect((x0-xc)/max(model.do,model.wo))

        T[i] = P0*overlap(x0,u0,Ek)**2
    return T
