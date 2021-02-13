import numpy as np
from . import SimulationOptions
from .AWG import *

class Simulate:
	"""
	Simulate entire AWG from input to output at given wavelength.

	INPUT:
		model  - AWG system to Simulate
		lmbda  - center wavelength [μm]
		_input - Number of input waveguide
	
	Optional:
		Points  - Number of point to sample over the differents fields
		Options - Using some custom simulation options using the SimulationOptions function

	OUTPUT:
		None

	ATTRIBUTE:
		inputField   - Field at the input waveguide
		arrayField   - Field at the end of the arrayed section
		outputField  - Field at the output waveguide
		transmission - Transmission for each AWG ouput channel
		lmbda        - Wavelenght use for the simulation
	"""
	def __init__(self,model,lmbda,_input = 0,**kwargs):

		_in = kwargs.keys()
		if "Points" in _in:
			points = kwargs["Points"]
		else:
			points = 250
		if "Options" in _in:
			Options = kwargs["Options"]
		else:
			Options = SimulationOptions()

		if Options.CustomInputField != []:
			F_iw = iw(model,lmbda,_input,Options.CustomInputField)
		else:
			F_iw = iw(model,lmbda,_input, ModeType = Options.ModeType, points = points)

		F_fpr1 = fpr1(model,lmbda,F_iw,points= points)

		F_aw = aw(model,lmbda,F_fpr1, ModeType = Options.ModeType,
				PhaseErrorVar = Options.PhaseErrorVariance, InsertionLoss = Options.InsertionLoss,
				PropagationLoss = Options.PropagationLoss)

		F_fpr2 = fpr2(model,lmbda,F_aw, points = points)

		F_ow = ow(model,lmbda,F_fpr2, ModeType = Options.ModeType)


		self.transmission = F_ow
		self.inputField = F_iw
		self.arrayField = F_aw
		self.outputField = F_fpr2
		self.lmbda = lmbda

