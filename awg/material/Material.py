"""
awg.material is a package for modeling material chromatic dispersion.
"""
import types
#import scipy as scp
from scipy.interpolate import Akima1DInterpolator
from . import *
from . import dispersion
from ..core import list_to_array


class Material:
	def __init__(self, model):

		if type(model) == list:
			model = list_to_array(model)
		if (str(type(model)) == "<class 'awg.material.Material.Material'>"):
			self.type = model.type
			self.model = model.model

		if type(model) == str:
			if model not in  "awg.material.Material.Material":
				model = ["awg.material", model]
			### model = str2func(model)

		if type(model) == types.FunctionType:
			try:
				pass # No python equivalent of this part in python
			finally:
				pass
			self.type = "function"
			self.model = model
		elif type(model) in [int, float]:
			self.type = "constant"
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
		""" 
		Return the index at a specific wavelength.

		lmbda - wavelenght [μm]
		T     - Température of the material [K] (optional)(def.295)

		"""

		if self.type == "constant":
			n = self.model
		elif self.type == "function":
			try:
				n = self.model(lmbda,T = T)
			except TypeError:
				n = self.model(lmbda)
		elif self.type == "polynomial":
			n = np.polyval(slef.model,lmbda) # To test
		elif self.type == "lookup":
			wavelength = self.model[:,0]
			index = self.model[:,1]
			n = Akima1DInterpolator(wavelength,index).__call__(lmbda,nu = 0,extrapolate = True) # Produce the Akima interpolation and extrapolation for unknow data of the lookup table
		return n

	def dispersion(self,lmbda1,lmbda2, point = 100):
		"""
		Return the dispersion relation between 2 wavelenght.

		lmdba1 - minimal wavelenght to consider [μm]
		lmbda2 - maximal wavelenght to consider [μm]
		point  - number of point to consider in the relation (optional)(def.100)

		"""
		return dispersion.dispersion(self.index, lmbda1, lmbda2, point = point)

	def groupindex(self,lmbda,T = 295):
		"""
		Return the group index at a specific wavelenght.

		lmbda - Wavelenght to consider [μm]
		T     - Temperature of the material [K] (optional)(def.295)

		"""		
		n0 = self.index(lmbda,T)
		n1 =self.index(lmbda-0.1,T)
		n2 = self.index(lmbda+0.1,T)

		return n0 - lmbda*(n2-n1)/0.2

	def groupDispersion(self,lmbda1,lmbda2, **kwargs):
		"""
		Return the group dispersion relation between 2 wavelenght.

		lmdba1 - minimal wavelenght to consider [μm]
		lmbda2 - maximal wavelenght to consider [μm]
		point  - number of point to consider in the relation (optional)(def.100)

		"""
		return dispersion.dispersion(self.groupindex,lmbda1,lmbda2,**kwargs)


