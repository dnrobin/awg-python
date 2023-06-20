from . import *
import numpy as np



def dispersion(model,lmbda1,lmbda2,**kwargs):
	"""
	Return the dispersion relation between 2 wavelenght.

	lmdba1 - minimal wavelenght to consider [μm]
	lmbda2 - maximal wavelenght to consider [μm]
	point  - number of point to consider in the relation (optional)(def.100)

	"""
	p = kwargs
	if "point" in p:
		if type(p["point"]) != int:
			raise ValueError("The number of point to use have to be an integer")

	else:
		p["point"] = 100
	lmbda = np.linspace(lmbda1,lmbda2,p["point"])

	if type(model) == types.FunctionType or (str(type(model)) == "<class 'method'>"):
		n = np.zeros(len(lmbda), dtype = object)
		if type(model(lmbda[0])) == list:
			"""In this case n will be an array of n list with each list representing a mode for all the wavelength """
			n_i = [model(i) for i in lmbda]

			z = np.zeros(len(n_i), dtype = int)

			for i in range(len(n_i)):
				z[i] = len(n_i[i])

			n = np.zeros(max(z), dtype = list)
			for i in range(len(n)):
				n[i] = []

			for item in n_i:
				while len(item) < max(z):
					item.append(0)
				for j in range(max(z)):
					n[j].append(item[j])
		else:
			for i in range(len(lmbda)):
				n[i] = model(lmbda[i])
	elif str(type(model)) in {
		"<class 'awg.material.Material.Material'>",
		"<class 'awg.Waveguide.Waveguide'>",
	}:
		n = np.zeros(np.shape(lmbda))
		for i in range(len(lmbda)):
			n[i] = model.index(lmbda[i])
	else:
		raise ValueError("Wrong model provided")
	return n, lmbda

