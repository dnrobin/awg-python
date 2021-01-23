from .core import *

class SimulationOptions:
	__slots__ = [
				"ModeType",
				"UseMagneticField",
				"InsertionLoss",
				"PropagationLoss",
				"PhaseErrorVariance"
				"CustumInputField"]

	def __init__(self,**kwargs):

		_in  = kwargs.keys()
		if "ModeType" in _in:
			ModeType = kwargs["ModeType"]
		else:
			ModeType = "gaussian"

		if ModeType.lower() not in ['rect','gaussian','solve']:
			raise ValueError("Mode type must be 'rect','gaussian'or 'solve'.")

		if "UseMagneticField" in _in:
			UseMagneticField = kwargs["UseMagneticField"]
		else:
			UseMagneticField = False
		
		if type(UseMagneticField) != bool:
			raise TypeError("UseMagneticField must be a boolean")

		if "InsertionLoss" in _in:
			InsertionLoss = kwargs["InsertionLoss"]
		else:
			InsertionLoss = 0

		if InsertionLoss < 0:
			raise ValueError("The insertion loss must be bigger or equal to 0")

		if "PropagationLoss" in _in:
			PropagationLoss = kwargs["PropagationLoss"]
		else:
			PropagationLoss = 0

		if PropagationLoss < 0:
			raise ValueError("The propagation loss must be bigger or equal to 0")

		if "PhaseErrorVariance" in _in:
			PhaseErrorVariance = kwargs["PhaseErrorVariance"]
		else:
			PhaseErrorVariance = 0

		if PhaseErrorVariance < 0:
			raise ValueError("The phase error variance must be bigger or equal to 0")

		if "CustumInputField" in _in:
			CustumInputField = kwargs["CustumInputField"]
		else:
			CustumInputField = []
