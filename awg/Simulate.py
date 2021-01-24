import numpy as np
from . import SimulationOptions
from .AWG import *

class Simulate:
	def __init__(self,model,lmbda,_input = 0,**kwargs):

		_in = kwargs.keys()
		if "points" in _in:
			points = kwargs["points"]
		else:
			points = 250
		if "Options" in _in:
			Options = kwargs["Options"]
		else:
			Options = SimulationOptions()

		if len(Options.CustomInputField) != 0:
			F_iw = AWG.iw(model,lmbda,_input,Options.CustomInputField)
		else:
			F_iw = iw(model,lmbda,_input, ModeType = Options.ModeType, points = points)

		F_fpr1 = fpr1(model,lmbda,F_iw,points= points)

		F_aw = aw(model,lmbda,F_fpr1,ModeType = Options.ModeType,
				PhaseErrorVar = Options.PhaseErrorVariance, InsertionLoss = Options.InsertionLoss,
				PropagationLoss = Options.PropagationLoss)

		F_fpr2 = fpr2(model,lmbda,F_aw, points = points)

		F_ow = ow(model,lmbda,F_fpr2, ModeType = Options.ModeType)

		self.transmission = F_ow
		self.inputField = F_iw
		self.arrayField = F_aw
		self.outputField = F_fpr2
		self.lmbda = lmbda

