import numpy as np
from . import SimulationOptions, Simulate
from .AWG import *

class Spectrum:
	"""
	Simulate entire AWG device over wavelength range and extract transmission

	INPUT:

		model     - AWG system to Simulate
		lmbda     - center wavelength [μm]
		bandwidth - badwidth use around the center wavelenght [μm]
	OPTIONAL :
		Points  - Number of point to sample over the calculated field (def.250)
		Samples - Number of point to sample over the bandwidth (def.100)
		Options - Using some custom simulation options using the SimulationOptions function
	OUTPUT:
		None
	
	ATTRIBUTE:
		wavelength   - Array of the wavelegth use for the simulation
		transmission - Array of transmission for each ouput channel of the AWG at every wavelenght

	"""
	def __init__(self,model,lmbda,bandwidth,**kwargs):
		_in = kwargs.keys()

		points = kwargs["Points"] if "Points" in _in else 250
		Samples = kwargs["Samples"] if "Samples" in _in else 100
		Options = kwargs["Options"] if "Options" in _in else SimulationOptions()
		wvl = lmbda + np.linspace(-0.5,0.5,Samples)*bandwidth

		T = np.zeros((Samples,model.No), dtype = complex)


		# Replacement for the wait bar
		for i in range(Samples):
			R = Simulate(model,wvl[i],Options = Options,points = points)
			T[i,:] = R.transmission
			print(f"{i+1}/{Samples}")

		self.wavelength = wvl
		self.transmission = T