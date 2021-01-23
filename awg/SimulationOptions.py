from .core import *

class SimulationOptions:
	__slots__ = [
				"ModeType",
				"UseMagneticField",
				"InsertionLoss",
				"PropagationLoss",
				"PhaseErrorVariance",
				"CustomInputField"]

	def __init__(self,**kwargs):

		_in  = kwargs.keys()
		if "ModeType" in _in:
			self.ModeType = kwargs["ModeType"]
		else:
			self.ModeType = "gaussian"

		if self.ModeType.lower() not in ['rect','gaussian','solve']:
			raise ValueError("Mode type must be 'rect','gaussian'or 'solve'.")

		if "UseMagneticField" in _in:
			self.UseMagneticField = kwargs["UseMagneticField"]
		else:
			self.UseMagneticField = False
		
		if type(self.UseMagneticField) != bool:
			raise TypeError("UseMagneticField must be a boolean")

		if "InsertionLoss" in _in:
			self.InsertionLoss = kwargs["InsertionLoss"]
		else:
			self.InsertionLoss = 0

		if self.InsertionLoss < 0:
			raise ValueError("The insertion loss must be bigger or equal to 0")

		if "PropagationLoss" in _in:
			self.PropagationLoss = kwargs["PropagationLoss"]
		else:
			self.PropagationLoss = 0

		if self.PropagationLoss < 0:
			raise ValueError("The propagation loss must be bigger or equal to 0")

		if "PhaseErrorVariance" in _in:
			self.PhaseErrorVariance = kwargs["PhaseErrorVariance"]
		else:
			self.PhaseErrorVariance = 0

		if self.PhaseErrorVariance < 0:
			raise ValueError("The phase error variance must be bigger or equal to 0")

		if "CustomInputField" in _in:
			self.CustomInputField = kwargs["CustomInputField"]
		else:
			self.CustomInputField = []
