"""
awg.material is a package for modeling material chromatic dispersion.
"""
import types
from .Material import * 
from . import dispersion


class Material:
	def __init__(self, model):
	
		if (str(type(model)) == "<class 'awg.material.Material'>"):
			self.type = model.type
			#self.model = model.model

		if type(model) == str:
			if model not in  "awg.material.Material":
				model = ["awg.material", model]
			### model = str2func(model)

		if type(model) == types.FunctionType:
			try:
				pass # No python equivalent of this part in python
			finally:
				pass
			self.type = "function"
		elif str(type(model)) == "<class 'numpy.ndarray'>":
			if np.size(model) == 1 :
				self.type = "constant"
			elif len(np.shape(model)) == 1:
				self.type = "polynomial"
			else:
				nr,nc = np.shape(model)
				if nc > nr:
					model = model.conj().T
				if np.shape(model)[1] > 2:
					raise ValueError("Invalid model argument provided for Material(<model>), data set must be a 2 column matrix with column 1 containing wavelength data and column 2 containing refractive index.")
			self.type = "lookup"
		self.model = model

	def index(self,lmbda,T = 295):
		"""Calculates refractive index at given wavelength and
		   temperature using lookup data or model equation. """
		if self.type == "constant":
			n = self.model
		elif self.type == "function":
			n = self.model(lmbda,T)
		elif self.type == "polynomial":
			n = np.polyval(slef.model,lmbda) # To test
		return n

	def dispersion(self,lmbda1,lmbda2):
		pass




