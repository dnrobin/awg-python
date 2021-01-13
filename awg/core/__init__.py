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


			C = -np.sqrt(nc**2/n0/neff[m]/(h+1/(p*p2) + 1/(q*q2)))
			Hm1 = np.cos(k*h/2 + f)*np.exp(p*(h/2 + y[i1]))
			Hm2 = np.cos(k*y[i2] - f)
			Hm3 = np.cos(k*h/2 - f)*np.exp(q*(h/2 - y[i3]))
			Hm = np.concatenate((Hm1,Hm2,Hm3))*C

			E[:,m,1] = -neff[m]*n0/n**2*Hm
			E[:,m,2] = 1j*n0/(k0*nc**2)*np.concatenate((np.zeros(1),np.diff(Hm)))
			H[:,m,1] = Hm
			#print(E[:,:,0],E[:,:,1],E[:,:,2])

	return E,H,y,neff


x = slabmode(1.5,1,SiO2,Si,SiO2, points = 10)
print(x)

#a = [10,9,8,7,6,5,4,3,2,1,0]
#for i, e in enumerate(a):
#	print(i,e)



def wgindex():
    pass

def wgmode():
    pass

def diffract():
    pass

def overlap():
    pass