from .core import *
from .material import *
from .material.Material import Material
from . import Field, Waveguide, Aperture
import types
from numpy.random import randn
from tabulate import tabulate

class AWG:
    """
         Arrayed Waveguide Grating Model
    
    PROPERTIES:
        lambda_c - design center wavelength
        clad - top cladding material
        core - core (guiding) material
        subs - bottom cladding material, note that materials can be assigned by a
            string literal refering to a awg.material.* function, a function handle
            for computing dipersion, a lookup table, a constant value or an
            awg.material.Material object instance. See awg.material.Material for
            details.
        w - waveguide core width
        h - waveguide code height
        t - waveguide slab thickness (for rib waveguides) (def. 0)
        N - number of arrayed waveguides
        m - diffraction order
        R - grating radius of carvature (focal length)
        g - gap width between array apertures
        d - array aperture spacing
        L0 - minimum waveguide length offset (def. 0)
        Ni - number of input waveguides
        wi - input waveguide aperture width
        di - input waveguide spacing (def. 0)
        li - input waveguide offset spacing (def. 0)
        No - number of output waveguides
        wo - output waveguide aperture width
        do - output waveguide spacing (def. 0)
        lo - output waveguide offset spacing (def. 0)
        defocus - added defocus to R (def. 0)
        confocal - use confocal arrangement rather than Rowland (def. false)
    
    CALCULATED PROPERTIES:
        wg - array waveguide aperture width
        dl - array length increment
    """
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
        '_confocal',  # use confocal arrangement rather than Rowland (def. false)
        '_defocus',   # radial defocus (def. 0)
        '_wa',        # waveguide aperture width
        '_Ri',        # input/output radius curvature # To add in future update
        '_Ra'         # array radius curvature # To add in future update
    ]
    def __init__(self,**kwargs):
        _in = kwargs.keys()

        if "lambda_c" in _in:
            if type(kwargs["lambda_c"]) in [float, int] and kwargs["lambda_c"] > 0:
                self._lambda_c = kwargs["lambda_c"]
            else:
                raise ValueError("The central wavelength [um] must be a positive float or integer.")
        else:
            self._lambda_c = 1.550

        if "clad" in _in:
            if type(kwargs["clad"]) in [types.FunctionType, float, int]:
                self._clad = Material(kwargs["clad"])
            elif (str(type(kwargs["clad"])) == "<class 'awg.material.Material.Material'>"):
                self._clad = kwargs["clad"]
            elif type(kwargs["clad"]) == list:
                self._clad = Material(list_to_array(kwargs["clad"]))
            elif str(type(kwargs["clad"])) == "<class 'numpy.ndarray'>":
                self._clad = Material(kwargs["clad"])
            else:
                raise ValueError("The cladding must be a material or a float representing its refractive index.")
        else:
            self._clad = Material(SiO2)

        if "core" in _in:
            if type(kwargs["core"]) in [types.FunctionType, float, int]:
                self._core = Material(kwargs["core"])
            elif (str(type(kwargs["core"])) == "<class 'awg.material.Material.Material'>"):
                self._core = kwargs["core"]
            elif type(kwargs["core"]) == list:
                self._core = Material(list_to_array(kwargs["core"]))
            elif str(type(kwargs["core"])) == "<class 'numpy.ndarray'>":
                self._core = Material(kwargs["core"])
            else:
                raise ValueError("The core must be a material or a float representing its refractive index.")
        else:
            self._core = Material(Si)


        if "subs" in _in:
            if type(kwargs["subs"]) in [types.FunctionType, float, int]:
                self._subs = Material(kwargs["subs"])
            elif (str(type(kwargs["subs"])) == "<class 'awg.material.Material.Material'>"):
                self._subs = kwargs["subs"]
            elif type(kwargs["subs"]) == list:
                self._subs = Material(list_to_array(kwargs["subs"]))
            elif str(type(kwargs["subs"])) == "<class 'numpy.ndarray'>":
                self._subs = Material(kwargs["subs"])
            else:
                raise ValueError("The substrate must be a material or a float representing its refractive index.")
        else:
            self._subs = Material(SiO2)

        if "w" in _in:
            if type(kwargs["w"]) in [int, float] and kwargs["w"] > 0:
                self._w = kwargs["w"]
            else: 
                raise ValueError("The array waveguide core width 'w' [um] must be positive and be a float or an integer.")
        else:
            self._w = 0.450

        if "h" in _in:
            if type(kwargs["h"]) in [int, float] and kwargs["h"] > 0:
                self._h = kwargs["h"]
            else: 
                raise ValueError("The array waveguide core height 'h' [um] must be positive and be a float or an integer.")
        else:
            self._h = 0.220

        if "t" in _in:
            if type(kwargs["t"]) in [int, float] and kwargs["t"] >= 0:
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
            if type(kwargs["m"]) in [int, float] and kwargs["m"] > 0:
                self._m = kwargs["m"]
            else: 
                raise ValueError("The order of diffraction 'm' must be a positive integer.")
        else:
            self._m = 30

        if "R" in _in:
            if type(kwargs["R"]) in [int, float] and kwargs["R"] > 0:
                self._R = kwargs["R"]
            else:
                raise ValueError("The grating radius of curvature (focal length) 'R' [um] must be a positive float or integer")
        else:
            self._R = 100

        if "d" in _in:
            if type(kwargs["d"]) in [int, float] and kwargs["d"] > 0:
                self._d = kwargs["d"]
            else:
                raise ValueError("The array aperture spacing 'd' [um] must be a positive float or integer")
        else:
            self._d = 1.3

        if "g" in _in:
            if type(kwargs["g"]) in [int, float] and kwargs["g"] > 0:
                self._g = kwargs["g"]
            else:
                raise ValueError("The gap width between array aperture 'g' [um] must be a positive float or integer")
        else:
            self._g = 0.2

        if "L0" in _in:
            if type(kwargs["L0"]) in [int, float] and kwargs["L0"] >= 0:
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
            if type(kwargs["wi"]) in [int, float] and kwargs["wi"] > 0:
                self._wi = kwargs["wi"]
            else:
                raise ValueError("The input waveguide aperture width 'wi' [um] must be a positive float or integer")
        else:
            self._wi = 1

        if "di" in _in:
            if type(kwargs["di"]) in [int, float] and kwargs["di"] > 0:
                self._di = kwargs["di"]
            else:
                raise ValueError("The input waveguide spacing 'di' [um] must be a positive float or integer")
        else:
            self._di = 0

        if "li" in _in:
            if type(kwargs["li"]) in [int, float] and kwargs["li"] >= 0:
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
            if type(kwargs["wo"]) in [int, float] and kwargs["wo"] > 0:
                self._wo = kwargs["wo"]
            else:
                raise ValueError("The output waveguide aperture width 'wo' [um] must be a positive float or integer")
        else:
            self._wo = 1

        if "do" in _in:
            if type(kwargs["do"]) in [int, float] and kwargs["do"] > 0:
                self._do = kwargs["do"]
            else:
                raise ValueError("The output waveguide spacing 'do' [um] must be a positive float or integer")
        else:
            self._do = 0

        if "lo" in _in:
            if type(kwargs["lo"]) in [int, float] and kwargs["lo"] >= 0:
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
            if type(kwargs["defocus"]) in [int, float] and kwargs["defocus"] > 0:
                self._defocus = kwargs["defocus"]
            else:
                raise ValueError("The radial defocus must be a positive float or integer")
        else:
            self._defocus = 0

        if "wa" in _in:
            if type(kwargs["wa"]) in [int, float] and kwargs["wa"] > 0:
                self._wa = kwargs["wa"]
            else:
                raise ValueError("The waveguide aperture width 'wa' [um] must be a positive float or integer")
        else:
            self._wa = self._d - self._g




    def getSlabWaveguide(self):
        """
        Return the slab waveguide propreties.
        """
        return Waveguide.Waveguide(clad = self._clad,core = self._core,subs = self._subs,h = self._h, t = self._h)
    
    def getArrayWaveguide(self):
        """
        Return the arrayed waveguide propreties.
        """
        return Waveguide.Waveguide(clad = self._clad,core = self._core,subs = self._subs,w = self._w,h = self._h, t = self._t)

    def getInputAperture(self):
        """
        Return the input waveguide aperture.
        """
        return Aperture.Aperture(clad = self._clad,core = self._core,subs = self._subs,w = self._wi,h = self._h)


    def getArrayAperture(self):
        """
        Return the slab waveguide propreties.
        """
        return Aperture.Aperture(clad = self._clad,core = self._core,subs = self._subs,w = self._wa,h = self._h)

    def getOutputAperture(self):
        """
        Return the slab waveguide propreties.
        """
        return Aperture.Aperture(clad = self._clad,core = self._core,subs = self._subs,w = self._wo,h = self._h)

    def __str__(self):
        
        if type(self._clad.model) == types.FunctionType:
            clad = self._clad.model.__name__
        elif self._clad.type == "constant":
            clad = self._clad.model
        elif self._clad.type == "polynomial":
            if len(self._clad.model) <= 3:
                clad = self._clad.model
            else:
                clad = f"[{self._clad.model[0]},...,{self._clad.model[-1]}]"
        elif self._clad.type == "lookup":
            clad = "lookup table"

        if type(self._core.model) == types.FunctionType:
            core = self._core.model.__name__
        elif self._core.type == "constant":
            core = self._core.model
        elif self._core.type == "polynomial":
            if len(self._core.model) <= 3:
                core = self._core.model
            else:
                core = f"[{self._core.model[0]},...,{self._core.model[-1]}]"
        elif self._core.type == "lookup":
            core = "lookup table"

        if type(self._subs.model) == types.FunctionType:
            subs = self._subs.model.__name__
        elif self._subs.type == "constant":
            subs = self._subs.model
        elif self._subs.type == "polynomial":
            if len(self._subs.model) <= 3:
                subs = self._subs.model
            else:
                subs = f"[{self._subs.model[0]},...,{self._subs.model[-1]}]"
        elif self._subs.type == "lookup":
            subs = "lookup table"
        
        return tabulate([['lambda_c', self._lambda_c, "Central wavelenght"], 
                        ['clad', clad , "clad material"], 
                        ['core', core, "core material"], 
                        ['subs', subs, "subs material"],
                        ['w', self.w,"core width [\u03BCm]"],
                        ['h', self._h,"core height [\u03BCm]"],
                        ["t",self.t,"slab thickess for rib waveguides [\u03BCm]"],
                        ["N",self.N,"Number of arrayed waveguide"],["m",self.m, "Diffraction order"],
                        ["R",self.R,"grating radius of carvature (focal length) [\u03BCm]"],
                        ["d",self.d,"array aperture spacing"],["g",self.g,"gap width between array apertures"],
                        ["L0",self.L0,"minimum waveguide length offset (def. 0) [\u03BCm]"],
                        ["Ni",self.Ni,"Number of input waveguide"],
                        ["wi",self.wi,"input waveguide aperture width [\u03BCm]"],
                        ["di",self.di,"input waveguide spacing (def. 0) [\u03BCm]"],
                        ["li",self.li,"input waveguide offset spacing (def. 0)"],
                        ["No",self.No,"Number of ouput waveguide"],
                        ["wo",self.wo,"ouput waveguide aperture width [\u03BCm]"],
                        ["do",self.do,"ouput waveguide spacing (def. 0) [\u03BCm]"],
                        ["lo",self.lo,"ouput waveguide offset spacing (def. 0)"],
                        ["confocal",self.confocal,"use confocal arrangement rather than Rowland (def. false)"],
                        ], headers=['parameters', 'Value', 'definition'])

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
        return self.getArrayWaveguide().index(self.lambda_c,1)[0]

    @property
    def ncore(self):
        return self.core.index(self.lambda_c)

    @property
    def nclad(self):
        return self.clad.index(self.lambda_c)
    
    @property
    def subs(self):
        return selfsubs.index(self.lambda_c)

    

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
        if type(w) in [int, float] and w > 0:
            self._w = w
        else:
            raise ValueError("The array waveguide core width 'w' [um] must be positive and be a float or an integer.")

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self,h):
        if type(h) in [int, float] and h > 0:
            self._h = h
        else:
            raise ValueError("The array waveguide core height 'h' [um] must be positive and be a float or an integer.")

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self,t):
        if type(t) in [int, float] and t >= 0:
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
        if type(m) in [int, float] and m > 0:
            self._m = m
        else:
            raise ValueError("The order of diffraction 'm' must be a positive integer.")
    
    @property
    def R(self):
        return self._R

    @R.setter
    def R(self,R):
        if type(R) in [int, float] and R > 0:
            self._R = R
        else:
            raise ValueError("The grating radius of curvature (focal length) 'R' [um] must be a positive float or integer")

    @property
    def d(self):
        return self._d

    @d.setter
    def d(self,d):
        if type(d) in [int, float] and d > 0:
            self._d = d
        else:
            raise ValueError("The array aperture spacing 'd' [um] must be a positive float or integer")

    @property
    def g(self):
        return self._g

    @g.setter
    def g(self,g):
        if type(g) in [int, float] and g > 0:
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
        if type(wi) in [int, float] and wi > 0:
            self._wi = wi
        else:
            raise ValueError("The input waveguide aperture width 'wi' [um] must be a positive float or integer")

    @property
    def di(self):
        return self._di

    @di.setter
    def di(self,di):
        if type(di) in [int, float] and di > 0:
            self._di = di
        else:
            raise ValueError("The input waveguide spacing 'di' [um] must be a positive float or integer")

    @property
    def li(self):
        return self._li

    @li.setter
    def li(self,li):
        if type(li) in [int, float] and li >= 0:
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
    def wo(self,wo):
        if type(wo) in [int, float] and wo > 0:
            self._wo = wo
        else:
            raise ValueError("The output waveguide aperture width 'wo' [um] must be a positive float or integer")

    @property
    def do(self):
        return self._do

    @do.setter
    def do(self,do):
        if type(do) in [int, float] and do > 0:
            self._do = do
        else:
            raise ValueError("The output waveguide spacing 'do' [um] must be a positive float or integer")

    @property
    def lo(self):
        return self._lo

    @lo.setter
    def lo(self,lo):
        if type(lo) in [int, float] and lo >= 0:
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
        if type(defocus) in [int, float] and defocus > 0:
            self._defocus = defocus
        else:
            raise ValueError("The defocus or R must be a positive float or integer")



    @property
    def wa(self):
        return self._wa

    @wa.setter
    def wa(self,wa):
        if type(wa) in [int, float] and wa > 0:
            self._wa = wa
        else:
            raise ValueError("The waveguide aperture width 'wa' [um] must be a positive float or integer")

    @property
    def dl(self):
        return self._m*self._lambda_c/self.nc

    @dl.setter
    def dl(self,dl):
        if type(dl) in [int, float] and dl > 0:
            self._dl = dl
        else:
            raise ValueError("The arrayed waveguide lenght increment 'dl' must be a positive float or integer")

    @property
    def wg(self):
        return self._d-self._g
    
    



def iw(model, lmbda, _input = 0, u = np.array([]),**kwargs):
    """
    Generates input waveguide field distribution.

    INPUT :
        model - AWG systeme
        lmbda - center wavelength [μm]
        u     - custom input field (def.[])

    OPTIONAL :
        ModeType - Type of mode to use (rect, gaussian,solve)(def.gaussian)
        points   - number of field sample
    OUTPUT :
        output field
    """

    _in = kwargs.keys()

    ModeType = kwargs["ModeType"] if "ModeType" in _in else "gaussian"
    if ModeType not in ["rect","gaussian", "solve"]:
        raise ValueError(f"Wrong mode type {ModeType}.")

    points = kwargs["points"] if "points" in _in else 100
    if str(type(u)) == "<class 'awg.Field.Field'>":
        F = u
    elif len(u) == 0:
        x = np.linspace(-1,1,points)*max(model.di,model.wi)
        F = model.getInputAperture().mode(lmbda, x= x, ModeType = ModeType)
    elif (min(u.shape) > 2) or (len(u.shape) > 2) :
        raise ValueError("Data provided for the input field must be a two column matrix of coordinate, value pairs.")
    else:
        n,m = u.shape
        F = Field.Field(u[:,0],u[:,1])


    return F.normalize()

def fpr1(model,lmbda,F0,**kwargs):
    """
    Propagates the field in the first free propagation region.

    INPUT :
        model - AWG systeme
        lmbda - center wavelength [μm]
        F0    - Input Field

    OPTIONAL :
        x      - spatial range of the field at the end of fpr
        points - number of field sample

    OUTPUT :
        Field at the end of the first fpr
    """
    _in =kwargs.keys()

    x = kwargs["x"] if "x" in _in else []
    _input = kwargs["Input"] if "Input" in _in else 0
    points = kwargs["points"] if "points" in _in else 250
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

    uf = diffract(lmbda/ns,up,xp,xf,zf)[0]


    return Field.Field(sf,uf).normalize(F0.power())

def aw(model,lmbda,F0,**kwargs):
    """
    Couples input field to the array apertures and propagates the fields
    along the waveguide array to the other end.

    INPUT :
        model - AWG systeme
        lmbda - center wavelength [μm]
        F0    - Input Field
    OPTIONAL:
        ModeType       - Type of mode to use (rect, gaussian,solve)(def.gaussian)
        PhaseError     - Amplitude of random phase error through arrayed waveguide
        InsertionLoss  - Insertion loss in the model [dB]
        PopagationLoss - Propagation loss [dB/cm]
    OUTPUT :
        Field at the end of the arrayed waveguide section
    """
    _in = kwargs.keys()

    ModeType = kwargs["ModeType"] if "ModeType" in _in else "gaussian"
    if ModeType.lower() not in ["rect","gaussian", "solve"]:
        raise ValueError(f"Wrong mode type {ModeType}.")

    PhaseErrorVar = kwargs["PhaseErrorVar"] if "PhaseErrorVar" in _in else 0
    InsertionLoss = kwargs["InsertionLoss"] if "InsertionLoss" in _in else 0
    PropagationLoss = kwargs["PropagationLoss"] if "PropagationLoss" in _in else 0
    x0 = F0.x
    u0 = F0.Ex
    P0 = F0.power()

    k0 = 2*np.pi/lmbda
    nc = model.getArrayWaveguide().index(lmbda,1)[0]

    pnoise = randn(1,model.N)[0]*np.sqrt(PhaseErrorVar)
    iloss = 10**(-abs(InsertionLoss)/10)

    Aperture = model.getArrayAperture()

    Ex = np.zeros(len(F0.E))

    for i in range(model.N):
        xc = (i - (model.N-1)/2)*model.d

        Fk =  Aperture.mode(lmbda,x = x0-xc, ModeType = ModeType).normalize()

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
    """
    Propagates the field in the second free propagation region.

    INPUT :
        model - AWG systeme
        lmbda - center wavelength [μm]
        F0    - Input Field

    OPTIONAL :
        x      - spatial range of the field at the end of fpr
        points - number of field sample

    OUTPUT :
        Field at the end of the second fpr
    """
    _in  = kwargs.keys()

    x = kwargs["x"] if "x" in _in else []
    points = kwargs["points"] if "points" in _in else 250
    x0 = F0.x
    u0 = F0.Ex

    ns = model.getSlabWaveguide().index(lmbda,1)[0]

    R = model.R
    r = R/2
    if model.confocal:
        r = R

    sf = np.linspace(-np.pi/2,np.pi/2,points)*r if len(x) == 0 else x
    a = x0/R
    xp = R*np.tan(a)
    dp = R*(1/np.cos(a))-R
    up = u0*np.exp(1j*2*np.pi/lmbda*ns*dp)

    a = sf/r
    xf = r*np.sin(a)
    zf = (model.defocus+R-r)+r*np.cos(a)

    uf = diffract(lmbda/ns,up,xp,xf,zf)[0]

    return Field.Field(sf,uf).normalize(F0.power())


def ow(model,lmbda,F0,**kwargs):
    """
    Compute output waveguide coupling

    INPUT :
        model - AWG systeme
        lmbda - center wavelength [μm]
        F0    - Field distribution at the beginning of the output waveguide

    OPTIONAL :
        ModeType - Type of mode to use (rect, gaussian,solve)(def.gaussian)
    OUTPUT :
        Power transmission for each output waveguide.
    """

    ModeType = kwargs.get("ModeType", "gaussian")
    if ModeType.lower() not in ["rect","gaussian", "solve"]:
        raise ValueError(f"Wrong mode type {ModeType}.")

    x0 = F0.x
    u0 = F0.Ex
    P0 = F0.power()

    Aperture = model.getOutputAperture()

    T = np.zeros(model.No, dtype = complex)

    for i in range(model.No):

        xc = model.lo +(i-(model.No-1)/2)*max(model.do,model.wo)

        Fk = Aperture.mode(lmbda,x = x0-xc, ModeType = ModeType)
        Ek = Fk.Ex

        Ek = Ek*rect((x0-xc)/max(model.do,model.wo))

        T[i] = P0*overlap(x0,u0,Ek)**2
    return abs(T)
