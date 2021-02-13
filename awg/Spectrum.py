import numpy as np
from . import SimulationOptions, Simulate
from .AWG import *

class Spectrum:
	def __init__(self,model,lmbda,bandwidth,**kwargs):
		_in = kwargs.keys()

		if "points" in _in:
			points = kwargs["points"]
		else:
			points = 250
		
		if "Samples" in _in:
			Samples = kwargs["Samples"]
		else:
			Samples = 100

		if "Options" in _in:
			Options = kwargs["Options"]
		else:
			Options = SimulationOptions()

		wvl = lmbda + np.linspace(-0.5,0.5,Samples)*bandwidth

		T = np.zeros((Samples,model.No), dtype = complex)

		
		# Replacement for the wait bar
		for i in range(Samples):
			R = Simulate(model,wvl[i],Options = Options,points = points)
			T[i,:] = R.transmission
			print(f"{i+1}/{Samples}")

		self.wavelength = wvl
		self.transmission = T