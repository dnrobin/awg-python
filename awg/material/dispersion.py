from . import *
from .. import Waveguide
import numpy as np



def dispersion(function,wvl1,wvl2,**kwargs):
	p = kwargs
	if "point" not in p.keys():
		p["point"] = 100
	else:
		if type(p["point"]) != int:
			raise ValueError("The number of point to use have to be an integer")

	lmbda = np.linspace(wvl1,wvl2,p["point"])

	if type(function) == types.FunctionType:
		n = np.zeros(len(lmbda))
		for i in len(lmbda):
			n[i] = function(lmbda[i])
	elif (str(type(function)) == "<class 'material.Material.Material'>") or (str(type(function)) == "<class 'Waveguide.Waveguide'>"):
		pass
		### Create Material and wavguide class before continuing

