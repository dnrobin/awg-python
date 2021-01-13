import sys, os
import types
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy.optimize import root
import cmath
sys.path.append(os.path.abspath(os.path.join('..')))
from material import *

def clamp(x,a,b):
	return min(max(x,a),b)

def step(x): # take only list or arrays
	return 1- np.double(x < 0)

def rect(x):
	return step(-x+1/2)*step(x+1/2)
### length of float and integer are not defined ###
def mat_prod(a, ma1,ma2):
	for i in range(len(a)):
		for j in range(len(a[i])):
			a[i][j] = ma1[i]*ma2[j]
	return a


def list_to_array(lst,dtype = complex):
	new_list = np.zeros(len(lst),dtype = dtype)
	for i,j in enumerate(lst):
		new_list[i] += j
	return new_list


def slabindex(lmbda0,t,na,nc,ns,**kwargs):
	""" Slabkwargsdex Guided mode effective index of planar waveguide.
	
	DESCRIPTION:
	Solves for the TE (or TM) effective index of a 3-layer slab waveguide
	           na          y
	   ^   ----------      |
	   t       nc          x -- z
	   v   ----------     
	           ns
	
	   with propagation in the +z direction

	 INPUT:
	 lambda0 - freespace wavelength
	 t  - core (guiding layer) thickness
	 na - cladding index (number|function)
	 nc - core index (number|function)
	 ns - substrate index (number|function)
	
	 OPTIONS:
	 Modes - max number of modes to solve
	 Polarisation - one of 'TE' or 'TM'
	
	 OUTPUT:
	
	 neff - vector of indexes of each supported mode
	
	 NOTE: it is possible to provide a function of the form n = @(lambda0) for 
	 the refractive index which will be called using lambda0."""
	
	neff = []

	if "Modes" not in kwargs.keys():
		kwargs["Modes"] = np.inf

	if "Polarisation" in kwargs.keys():
		if kwargs["Polarisation"] in ["TE","te","TM","tm"]:
			pass
	else:
		kwargs["Polarisation"] = "TE"


	if (type(na) == types.FunctionType) or (str(type(na)) == "<class 'material.Material.Material'>"):
		na = na(lmbda0)
	if (type(nc) == types.FunctionType) or (str(type(nc)) == "<class 'material.Material.Material'>"):
		nc = nc(lmbda0)
	if (type(ns) == types.FunctionType) or (str(type(ns)) == "<class 'material.Material.Material'>"):
		ns = ns(lmbda0)

	a0 = max(np.arcsin(ns/nc),np.arcsin(na/nc))
	if np.imag(a0) != 0:
		return neff

	if kwargs["Polarisation"].upper() == "TE":
		B1 = lambda a : np.sqrt(((ns/nc)**2 - np.sin(a)**2)+0j)
		r1 = lambda a : (np.cos(a)-B1(a))/(np.cos(a)+B1(a))
		
		B2 = lambda a : np.sqrt(((na/nc)**2 - np.sin(a)**2)+0j)
		r2 = lambda a : (np.cos(a)-B2(a))/(np.cos(a)+B2(a))

		phi1 = lambda a : np.angle(r1(a))
		phi2 = lambda a : np.angle(r2(a))

		M = math.floor((4*np.pi*t*nc/lmbda0*np.cos(a0)+phi1(a0) + phi2(a0))/(2*np.pi))
		
		for m in range(min(kwargs["Modes"],M+1)):
			a = root(lambda a : 4*np.pi*t*nc/lmbda0*np.cos(a)+phi1(a)+phi2(a)-2*(m)*np.pi,1)
			neff.append((np.sin(a.x)*nc)[0])
		return neff
	else:
		B1 = lambda a : (nc/ns)**2*np.sqrt(((ns/nc)**2 - np.sin(a)**2)+0j)
		r1 = lambda a : (np.cos(a)-B1(a))/(np.cos(a)+B1(a))
		
		B2 = lambda a : (nc/na)**2*np.sqrt(((na/nc)**2 - np.sin(a)**2)+0j)
		r2 = lambda a : (np.cos(a)-B2(a))/(np.cos(a)+B2(a))

		phi1 = lambda a : np.angle(r1(a))
		phi2 = lambda a : np.angle(r2(a))

		M = math.floor((4*np.pi*t*nc/lmbda0*np.cos(a0)+phi1(a0) + phi2(a0))/(2*np.pi))
		
		for m in range(min(kwargs["Modes"],M+1)):
			a = root(lambda a : 4*np.pi*t*nc/lmbda0*np.cos(a)+phi1(a)+phi2(a)-2*(m)*np.pi,1)
			neff.append((np.sin(a.x)*nc)[0])
		return neff		



def slabmode(lmbda0,t,na,nc,ns,**kwargs):
	"""Slab_mode  Guided mode electromagnetic fields of the planar waveguide.
	
	 DESCRIPTION:
	   solves for the TE (or TM) mode fields of a 3-layer planar waveguide
	
	           na          y
	   ^   ----------      |
	   t       nc          x -- z
	   v   ----------     
	           ns
	
	   with propagation in the +z direction

	 INPUT:
	 lambda0   - simulation wavelength (freespace)
	 t         - core (guiding layer) thickness
	 na        - top cladding index (number|function)
	 nc        - core layer index (number|function)
	 ns        - substrate layer index (number|function)
	 y (optional) - provide the coordinate vector to use
	
	 OPTIONS:
	 Modes - max number of modes to solve
	 Polarisation - one of 'TE' or 'TM'
	 Limits - coordinate range [min,max] (if y was not provided)
	 Points - number of coordinate points (if y was not provided)
	
	 OUTPUT:
	 y - coordinate vector
	 E,H - all x,y,z field components, ex. E(<y>,<m>,<i>), where m is the mode
	   number, i is the field component index such that 1: x, 2: y, 3:z
	
	 NOTE: it is possible to provide a function of the form n = @(lambda0) for 
	 the refractive index which will be called using lambda0."""

	n0 = 120*np.pi
	
	_in = kwargs
	if "y" not in _in.keys():
		_in["y"] = []

	if "Modes" not in kwargs.keys():
		kwargs["Modes"] = np.inf
	if "Polarisation" in _in.keys():
		if _in["Polarisation"] in ["TE","te","TM","tm"]:
			pass
	else:
		_in["Polarisation"] = "TE"

	if "Range" not in _in.keys():
		_in["Range"] = [-3*t,3*t]
	if "points" not in _in.keys():
		_in["points"] = 100

	if (type(na) == types.FunctionType) or (str(type(na)) == "<class 'material.Material.Material'>"):
		na = na(lmbda0)
	if (type(nc) == types.FunctionType) or (str(type(nc)) == "<class 'material.Material.Material'>"):
		nc = nc(lmbda0)
	if (type(ns) == types.FunctionType) or (str(type(ns)) == "<class 'material.Material.Material'>"):
		ns = ns(lmbda0)

	if _in["y"] == []:
		y = np.linspace(_in["Range"][0],_in["Range"][1],_in["points"])
	else:
		y = _in["y"]
	

	i1 = []
	i2 = []
	i3 = []
	for i,e in enumerate(y):
		if e < -t/2:
			i1.append(i)
		elif e <= t/2 and y[i] >= -t/2:
			i2.append(i)
		else:
			i3.append(i)

	neff = slabindex(lmbda0,t,ns,nc,na,Modes = _in["Modes"],Polarisation = _in["Polarisation"])
	E = np.zeros((len(y), len(neff), 3), dtype=complex)
	H = np.zeros((len(y), len(neff), 3), dtype=complex)
	k0 = 2*np.pi/lmbda0
	for m in range(len(neff)):
		p = k0*np.sqrt(neff[m]**2 - ns**2)
		k = k0*np.sqrt(nc**2 - neff[m]**2)
		q = k0*np.sqrt(neff[m]**2 - na**2)
		
		if _in["Polarisation"].upper() == "TE":
			
			f = 0.5*np.arctan2(k*(p - q),(k**2 + p*q))

			C = np.sqrt(n0/neff[m]/(t + 1/p + 1/q))

			Em1 = np.cos(k*t/2 + f)*np.exp(p*(t/2 + y[i1]))
			Em2 = np.cos(k*y[i2] - f)
			Em3 = np.cos(k*t/2 - f)*np.exp(q*(t/2 - y[i3]))
			Em = np.concatenate((Em1,Em2,Em3))*C
			

			H[:,m,1] = neff[m]/n0*Em
			H[:,m,2] = 1j/(k0*n0)*np.concatenate((np.zeros(1),np.diff(Em)))
			E[:,m,0] = Em
		else:
			n = np.ones(len(y))
			n[i1] = ns
			n[i2] = nc
			n[i3] = na

			f = 0.5*np.arctan2((k/nc**2)*(p/ns**2 - q/na**2),((k/nc**2)**2 + p/ns**2*q/na**2))
			p2 = neff[m]**2/nc**2 + neff[m]**2/ns**2 - 1
			q2 = neff[m]**2/nc**2 + neff[m]**2/na**2 - 1


			C = -np.sqrt(nc**2/n0/neff[m]/(t+1/(p*p2) + 1/(q*q2)))
			Hm1 = np.cos(k*t/2 + f)*np.exp(p*(t/2 + y[i1]))
			Hm2 = np.cos(k*y[i2] - f)
			Hm3 = np.cos(k*t/2 - f)*np.exp(q*(t/2 - y[i3]))
			Hm = np.concatenate((Hm1,Hm2,Hm3))*C

			E[:,m,1] = -neff[m]*n0/n**2*Hm
			E[:,m,2] = -1j*n0/(k0*nc**2)*np.concatenate((np.zeros(1),np.diff(Hm)))
			H[:,m,0] = Hm


	return E,H,y,neff



def wgindex(lmbda0,w,h,t,na,nc,ns,**kwargs):
	"""Effective index method for guided modes in arbitrary waveguide
	
	 DESCRIPTION:
	   solves for the TE (or TM) effective index of an etched waveguide
	   structure using the effectice index method.
	
	 USAGE:
	   - get effective index for supported TE-like modes:
	   neff = eim_index(1.55, 0.5, 0.22, 0.09, 1, 3.47, 1.44)
	
	              |<   w   >|
	               _________           _____
	              |         |            ^
	  ___    _____|         |_____ 
	   ^                                 h
	   t                                  
	  _v_    _____________________     __v__
	
	          II  |    I    |  II
	
	 INPUT:
	 lambda0   - free-space wavelength
	 w         - core width
	 h         - slab thickness
	 t         - slab thickness
	               t < h  : rib waveguide
	               t == 0 : rectangular waveguide w x h
	               t == h : uniform slab of thickness t
	 na        - (top) oxide cladding layer material index
	 nc        - (middle) core layer material index
	 ns        - (bottom) substrate layer material index
	
	 OPTIONS:
	 Modes - number of modes to solve
	 Polarisation - one of 'TE' or 'TM'
	
	 OUTPUT:
	 neff - TE (or TM) mode index (array of index if multimode)
	
	 NOTE: it is possible to provide a function of the form n = material(lambda0) for 
	 the refraction index which will be called using lambda0. """


	_in = kwargs
	if "Modes" not in _in.keys():
		_in["Modes"] = np.inf

	if "Polarisation" in _in.keys():
		if _in["Polarisation"] in ["TE","te","TM","tm"]:
			pass
	else:
		_in["Polarisation"] = "TE"

	if (type(na) == types.FunctionType) or (str(type(na)) == "<class 'material.Material.Material'>"):
		na = na(lmbda0)
	if (type(nc) == types.FunctionType) or (str(type(nc)) == "<class 'material.Material.Material'>"):
		nc = nc(lmbda0)
	if (type(ns) == types.FunctionType) or (str(type(ns)) == "<class 'material.Material.Material'>"):
		ns = ns(lmbda0)

	t = clamp(t,0,h)

	neff_I = slabindex(lmbda0,h,na,nc,ns,Modes = _in["Modes"], Polarisation = _in["Polarisation"])

	if t == h:
		neff = neff_I
		return neff

	if t > 0:
		neff_II = slabindex(lmbda0,t,na,nc,ns,Modes = _in["Modes"],Polarisation = _in["Polarisation"])
	else:
		neff_II = na
	
	neff = []

	if _in["Polarisation"].upper() in "TE":

		for m in range(min(len(neff_I),len(neff_II))):
			n = slabindex(lmbda0,w,neff_II[m],neff_I[m],neff_II[m],Modes = _in["Modes"],Polarisation = "TM")
			for i in n:
				if i > max(ns,na):
					neff.append(i)
	else:
		for m in range(min(len(neff_I),len(neff_II))):
			n = slabindex(lmbda0,w,neff_II[m],neff_I[m],neff_II[m],Modes = _in["Modes"],Polarisation = "TE")
			for i in n:
				if i > max(ns,na):
					neff.append(i)
	return neff


def wgmode(lmbda0,w,h,t,na,nc,ns,**kwargs):
	"""	eim_mode   Solve 2D waveguide cross section by effective index method.
	
	 This function solves for the fundamental TE (or TM) mode fields using 
	 effective index method.
	
	              |<   w   >|
	               _________           _____
	              |         |            ^
	  ___    _____|         |_____ 
	   ^                                 h
	   t                                  
	  _v_    _____________________     __v__
	
	          II  |    I    |  II
	
	 INPUT:
	 lambda    - free space wavelength
	 w         - core width
	 h         - core thickness
	 t         - slab thickness
	               t < h  : rib waveguide
	               t == 0 : rectangular waveguide w x h
	               t == h : uniform slab of thickness t
	 na        - (top) oxide cladding layer index of refraction
	 nc        - (middle) core layer index of refraction
	 ns        - (bottom) substrate layer index of refraction
	 x (optional) - provide the x coordinate vectors
	
	 OPTIONS:
	 Polarisation - one of 'TE' or 'TM'
	 Limits - limits for autogenerated coordinates
	 Points - number of points for autogenerated coordinates
	
	 OUTPUT:
	 E, H  - cell array of x, y and z field components such that E = {Ex, Ey, Ez}.
	 x     - coordinate vector
	 neff  - effective index of the modes solved
	
	 NOTE: it is possible to provide a function of the form n = @(lambda0) for 
	 the refraction index which will be called using lambda0. """

	t = clamp(t,0,h)

	_in = kwargs
	
	if "x" not in _in.keys():
		_in["x"] = []

	if "Polarisation" in _in.keys():
		if _in["Polarisation"] in ["TE","te","TM","tm"]:
			pass
	else:
		_in["Polarisation"] = "TE"
	
	if "XRange" not in _in.keys():
		_in["XRange"] = [-3*w,3*w]

	if "Sample" not in _in.keys():
		_in["Sample"] = 100

	if (type(na) == types.FunctionType) or (str(type(na)) == "<class 'material.Material.Material'>"):
		na = na(lmbda0)
	if (type(nc) == types.FunctionType) or (str(type(nc)) == "<class 'material.Material.Material'>"):
		nc = nc(lmbda0)
	if (type(ns) == types.FunctionType) or (str(type(ns)) == "<class 'material.Material.Material'>"):
		ns = ns(lmbda0)

	if _in["x"] == []:
		x = np.linspace(_in["XRange"][0],_in["XRange"][1],_in["Sample"])
	else:
		x = _in["x"]

	ni1 = np.zeros(len(x))

	if _in["Polarisation"].upper() in "TE":
		neff = wgindex(lmbda0,w,h,t,na,nc,ns,Polarisation = "TE",Modes = 1)[0]

		n_I = slabindex(lmbda0,h,na,nc,ns,Polarisation = "TE",Modes = 1)[0]
		
		if t > 0 :
			n_II = slabindex(lmbda0,t,na,nc,ns,Polarisation = "TE",Modes = 1)[0]
		else:
			n_II = na

		[Ek,Hk,_,_] = slabmode(lmbda0,w,n_II,n_I,n_II, Polarisation = "TM")
		#print(Hk)
		f = max(Ek[:,0,1])

		Ex = Ek[:,0,1]/f
		Hy = -Hk[:,0,0]/f
		Hz = -Hk[:,0,2]/f

		E = (Ex,ni1,ni1)
		H = (ni1,Hy,Hz)
	
	else:
		neff = wgindex(lmbda0,w,h,t,na,nc,ns,Polarisation = "TM",Modes = 1)[0]

		n_I = slabindex(lmbda0,h,na,nc,ns,Polarisation = "TM",Modes = 1)[0]
		
		if t > 0 :
			n_II = slabindex(lmbda0,t,na,nc,ns,Polarisation = "TM",Modes = 1)[0]
		else:
			n_II = na

		[Ek,Hk,_,_] = slabmode(lmbda0,w,n_II,n_I,n_II, Polarisation = "TE")
		#print(Hk)

		Ey = Ek[:,0,0]
		Ez = Ek[:,0,2]
		Hx = -Hk[:,0,1]

		E = (ni1,Ey,Ez)
		H = (Hx,ni1,ni1)


	return E,H,x,neff




def diffract(lmbda0,ui,xi,xf,zf, method = "rayleigh"):
	"""	DIFFRACT   1-D propagation using diffraction integral.
	
	   u = DIFFRACT(lambda, ui, xi, xf, zf) Numerically solves the one 
	   dimensional diffraction integral for propagation to the output 
	   coordinate(s) given by (xf,zf) from the input plane given by (xi,0)
	   with initial field distribution ui. The incoming light wave vector is 
	   assumed to be aligned with z-axis and the traveling wave is described 
	   by the retarded phase picture exp(-jkz).
	
	   u = DIFFRACT(..., METHOD) specifies which integral definition to use.
	   The choices are:
	       'rayleigh'  - (default) general purpose Rayleigh-Sommerfeld integral
	       'fresnel'   - Fresnel-Kirchoff approximation."""

	if (type(zf) == int) or (type(zf) == float) or (len(zf) == 1):
		zf = zf*np.ones(len(xf),dtype = complex)
	elif len(zf) != len(xf):
		raise ValueError("Coordinate vectors xf and zf must be the same length.")

	if type(ui) == list:
		ui = list_to_array(ui)
	if type(xi) == list:
		xi = list_to_array(xi)
	if type(xf) == list:
		xf = list_to_array(xf)

	k = 2*np.pi/lmbda0

	uf = np.zeros(len(xf),dtype = complex)

	for i in range(len(xf)):
		r = np.sqrt((xf[i]-xi)**2+zf[i]**2)
		if method == "rayleigh":

			uf[i] = np.sqrt(k/(2j*np.pi))*np.trapz(ui*zf[i]/r**(3/2)*np.exp(-1j*k*r),xi)

		elif method == "fresnel":

			uf[i] = np.sqrt(1j/(lmbda0*zf[i]))*np.exp(-1j*k*zf[i])*np.trapz(ui*np.exp(-1j*k/(2*zf[i])*(xi-xf[i])**2),xi)
		
		else:
			raise ValueError(f"Unrecognized {method} method.")

	return uf

#print(diffract(1.5,[0,1,2,3,2,1,0],[-3,-2,-1,0,1,2,3],[-5,-4,-3,-2,-1,0,1,2,3,4,5],[1.5], method = "fresnel"))

def overlap():
    pass