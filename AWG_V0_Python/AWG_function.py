import types
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy.optimize import root
from Materials import *
import cmath


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
def rsdi(lmbda,u0,x0,z,x = []):
	""""Rayleigh-Sommerfeld diffraction integral"""
	if x == []:
		x = x0+0j
	try:
		if len(z) == 1:
			z = z*np.ones(len(x))
		elif len(z) != len(x):
			raise ValueError("coordinate vectors x and z must be the same length!")
		k = 2*np.pi/lmbda
		u = np.zeros(len(x),dtype = complex)

	finally :
		for	i in range(len(x)):
			r = ((x0[:]*-1+x[i])**2 + z[i]**2)**0.5
			u[i] = z[i]/(lmbda*1j)*np.trapz(u0[:]*(1/k-r*1j)*np.exp(-1j*k*r)/(r**3),x0)
		return u,x

def overlap(x,u,v,Hu = None,Hv = None):
	uu = np.trapz((np.conj(u[:])*u[:]),x) #+ np.trapz(np.imag(np.conj(u[:])*u[:]),x)*1j
	vv = np.trapz((np.conj(v[:])*v[:]),x) #+ np.trapz(np.imag(np.conj(v[:])*v[:]),x)*1j
	uv = np.trapz((np.conj(u[:])*v[:]),x) #+ np.trapz(np.imag(np.conj(u[:])*v[:]),x)*1j
	#print(np.absolute(uv)**2/(uu*vv))

	return np.absolute(uv)**2/(uu*vv)



def slab_index(lmbda0,h,na,nc,ns,**kwargs):
	_in = kwargs
	if "N" not in _in.keys():
		_in["N"] = np.inf

	if "Mode" not in _in.keys():
		_in["Mode"] = "TE"
	if type(na) == types.FunctionType:
		na = na(lmbda0)
	if type(nc) == types.FunctionType:
		nc = nc(lmbda0)
	if type(ns) == types.FunctionType:
		ns = ns(lmbda0)

	a0 = max(np.arcsin(ns/nc),np.arcsin(na/nc))
	if np.iscomplex(a0):
		pass
	if _in["Mode"] in ["TE","te"]:
		B1 = lambda a : np.sqrt(((ns/nc)**2 - np.sin(a)**2)+0j)
		r1 = lambda a : (np.cos(a)-B1(a))/(np.cos(a)+B1(a))

		B2 = lambda a : np.sqrt(((na/nc)**2 - np.sin(a)**2)+0j)
	else:
		B1 = lambda a : (nc/ns)**2*np.sqrt(((ns/nc)**2 - np.sin(a)**2)+0j)
		r1 = lambda a : (np.cos(a)-B1(a))/(np.cos(a)+B1(a))

		B2 = lambda a : (nc/na)**2*np.sqrt(((na/nc)**2 - np.sin(a)**2)+0j)		

	r2 = lambda a : (np.cos(a)-B2(a))/(np.cos(a)+B2(a))

	phi1 = lambda a : np.angle(r1(a))
	phi2 = lambda a : np.angle(r2(a))

	M = math.floor((4*np.pi*h*nc/lmbda0*np.cos(a0)+phi1(a0) + phi2(a0))/(2*np.pi))
	neff = np.zeros(min(_in["N"],M+1))
	for m in range(min(_in["N"],M+1)):
		a = root(lambda t : 4*np.pi*h*nc/lmbda0*np.cos(t)+phi1(t)+phi2(t)-2*(m)*np.pi,1)
		neff[m] = np.sin(a.x)*nc
	return neff		

def eim_index(lmbda0,w,h,e,na,nc,ns,**kwargs):
	_in = kwargs
	if "N" not in _in.keys():
		_in["N"] = np.inf

	if "Mode" not in _in.keys():
		_in["Mode"] = "TE"
	if type(na) == types.FunctionType:
		na = na(lmbda0)
	if type(nc) == types.FunctionType:
		nc = nc(lmbda0)
	if type(ns) == types.FunctionType:
		ns = ns(lmbda0)

	e = clamp(e,0,h)

	neff_I = slab_index(lmbda0,h,na,nc,ns,N = _in["N"], Mode = _in["Mode"])

	if e == 0:
		return neff_I
	if e < h:
		neff_II = slab_index(lmbda0,h-e,na,nc,ns,N = _in["N"],Mode = _in["Mode"])
	else:
		neff_II = na
	neff = []

	if _in["Mode"] in ["TE","te"]:
		if type(neff_I) == float:
			neff_I = [neff_I]
		if type(neff_II) == float:
			neff_II = [neff_II]
		for m in range(min(len(neff_I),len(neff_II))):
			n = slab_index(lmbda0,w,neff_II[m],neff_I[m],neff_II[m],N = _in["N"],Mode = "TM")
			neff.extend(i for i in n if i > max(ns,na))
	else:
		for m in range(min(len(neff_I),len(neff_II))):
			n = slab_index(lmbda0,w,neff_II[m],neff_I[m],neff_II[m],N = _in["N"],Mode = "TE")
			neff.extend(i for i in n if i > max(ns,na))

	return neff

def slab_mode(lmbda0,h,na,nc,ns,**kwargs):
	n0 = 120*np.pi

	_in = kwargs
	if "y" not in _in.keys():
		_in["y"] = []

	if "Mode" not in _in.keys():
		_in["Mode"] = "TE"

	if "Range" not in _in.keys():
		_in["Range"] = [-3*h,3*h]
	if "sample" not in _in.keys():
		_in["sample"] = 100

	if type(na) == types.FunctionType:
		na = na(lmbda0)
	if type(nc) == types.FunctionType:
		nc = nc(lmbda0)
	if type(ns) == types.FunctionType:
		ns = ns(lmbda0)

	if _in["y"] == []:
		y = np.linspace(_in["Range"][0],_in["Range"][1],_in["sample"])
	else:
		y = _in["y"]
	i1 = []
	i2 = []
	i3 = []
	for i in range(len(y)):
		if y[i] < -h/2:
			i1.append(i)
		elif y[i] <= h / 2:
			i2.append(i)
		else:
			i3.append(i)

	neff = slab_index(lmbda0,h,ns,nc,na,Mode = _in["Mode"])
	E = np.zeros((len(y), len(neff), 3), dtype=complex)
	H = np.zeros((len(y), len(neff), 3), dtype=complex)
	k0 = 2*np.pi/lmbda0
	for m in range(len(neff)):
		p = k0*np.sqrt(neff[m]**2 - ns**2)
		k = k0*np.sqrt(nc**2 - neff[m]**2)
		q = k0*np.sqrt(neff[m]**2 - na**2)
		if _in["Mode"] in ["TE", "te"]:
			f = 0.5*np.arctan2(k*(p - q),(k**2 + p*q))

			C = np.sqrt(n0/neff[m]/(h + 1/p + 1/q))

			Em1 = np.cos(k*h/2 + f)*np.exp(p*(h/2 + y[i1]))
			Em2 = np.cos(k*y[i2] - f)
			Em3 = np.cos(k*h/2 - f)*np.exp(q*(h/2 - y[i3]))
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

	return y,E,H

def eim_mode(lmbda0,w,h,e,na,nc,ns,**kwargs):
	e = clamp(e,0,h)

	_in = kwargs

	if "x" not in _in.keys():
		_in["x"] = []
	if "y" not in _in.keys():
		_in["y"] = []

	if "Mode" not in _in.keys():
		_in["Mode"] = "TE"

	if "XRange" not in _in.keys():
		_in["XRange"] = [-3*w,3*w]
	if "YRange" not in _in.keys():
		_in["YRange"] = [-3*h,3*h]
	if "Sample" not in _in.keys():
		_in["Sample"] = 100

	if type(na) == types.FunctionType:
		na = na(lmbda0)
	if type(nc) == types.FunctionType:
		nc = nc(lmbda0)
	if type(ns) == types.FunctionType:
		ns = ns(lmbda0)

	if _in["x"] == []:
		x = np.linspace(_in["XRange"][0],_in["XRange"][1],_in["Sample"])
	elif _in["y"] == []:
		raise ValueError("x-coordinates were provided but not y.")
	else:
		x = _in["x"]

	Nx = len(x)
	dx = x[1]-x[0]

	y = (
		np.linspace(_in["YRange"][0], _in["YRange"][-1], _in["Sample"])
		if _in["y"] == []
		else _in["y"]
	)
	dy = y[1]-y[0]
	if _in["Mode"] in ["TE", "te"]:
		neff = eim_index(lmbda0,w,h,e,na,nc,ns,Mode = "TE",N = 1)

		[ _ , E_I,H_I] = slab_mode(lmbda0,h,na,nc,ns, y = y, Mode = "TE")
		n_I = slab_index(lmbda0,h,na,nc,ns,N = 1)
		n_II = slab_index(lmbda0,h-e,na,nc,ns,N = 1) if e < h else na
		[ _ ,E_III,H_III] = slab_mode(lmbda0, w, n_II, n_I, n_II, y = x, Mode = 'TM')
		Ny = len(y)
		E = np.zeros((Nx, Ny, 3),dtype = complex)
		H = np.zeros((Nx, Ny, 3),dtype = complex)
		#print(np.shape(E_I[:,0,0]),np.shape(E_III[:,0,1]),np.shape(E))
		#print(np.shape(E_I),np.shape(E_III),np.shape(E))
		E[:,:,0] = mat_prod(E[:,:,0],E_III[:,0,1],E_I[:,0,0].conj())#E_I[:,0,0]*E_III[:,0,1]
		#print(np.shape(E), np.shape(E_I), np.shape(E_III))
		#print(E[:,:,0])
		E[:,:,1] = mat_prod(E[:,:,1],E_III[:,0,0],E_I[:,0,1].conj())#
		E[:,:,2] = mat_prod(E[:,:,2],E_III[:,0,2],E_I[:,0,2].conj())#
		#print(E_III)
		H[:,:,0] = mat_prod(H[:,:,0],H_III[:,0,1],H_I[:,0,0].conj())#H_I[:,0,0]*H_III[:,0,1]
		H[:,:,1] = mat_prod(H[:,:,1],H_III[:,0,0],H_I[:,0,1].conj())#H_I[:,0,1]*H_III[:,0,0]
		H[:,:,2] = mat_prod(H[:,:,2],H_III[:,0,2],H_I[:,0,2].conj())#H_I[:,0,2]*H_III[:,0,2]

		return x,y,E,H,neff

	### TO DO --> MODE TM



#print(np.diff(np.array([0,1,2,3,4,4.5,5,10])))

