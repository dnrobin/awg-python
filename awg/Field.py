from .core import *
import numpy as np
from tabulate import tabulate

def DataFormat(D,sz):
	if len(D) == 0:
		D = np.zeros(sz, dtype = complex)[0]
		return D
	
	shape_D = [1]
	
	if np.shape(D)[0]> 1:
		try:
			shape_D = [len(i) for i in D]
		except TypeError:
			shape_D.append(len(D))
	else:
		shape_D.append(np.shape(D)[0])
	
	s =[]
	
	for i in sz:
		if i > 1:
			s.append(True)
		else:
			s.append(False)

	if False not in s:
		if shape_D != sz:
			raise ValueError("Wrong data format. The field must contain the same number of rows as the y-coordinate points and the same number of columns as the x-coordinate points.")
		else:
			if len(D) != max(sz):
				raise ValueError("Wrong data format. Expecting field data to be the same size as the coordinate elements.")
	
	return D

class Field:
	"""
	INPUT:

	X - Coordinate data, one of:
		vector  - A one dimensional vector of x coordinates.
		cell	- A cell array containing x and y coordinates: {x, y}.
			Note: for representing only y coordinates, pass a
			cell array of the form {[], y}.

	E - Electric field data.
	H - (optional) magnetic field data. Both of these fields can be 
		one of:
			vector  - A one dimensional vector of data points. In this
				case, the data will be mapped to the x-component 
				of the field by default.
			cell	- A cell array containing x, y and z component data
				in the form: {Ux, Uy, Uz}. For omitting data of
				certain components, use empty arrays. For example
				H = {[],Hy,[]}.
	"""
	def __init__(self,X, E, H = []):
		self.scalar = True
		if len(X) < 1:
			raise ValueError("At least one coordinate vector must be provided.")

		self._y = []
		self.dimens = 1
		
		if type(X) == tuple:
			self._x = X[0]

			if len(X) > 1 :
				self._y = X[1]
				self.dimens = 3
				if len(self._x) == 0:
					self.dimens = 2
				if len(self._y) == 0:
					self.dimens = 1

		else:
			if len([i for i in np.shape(X)]) > 1:
				raise ValueError("Wrong coordinate format. Must be a 1-D vector.")
			self._x = X

		if (len(self._x) == 0) and (len(self._y) == 0):
			raise Error("At least one coordinate vector must be provided.")
		
		if False in np.isreal(self._x) or False in np.isreal(self._y):
			raise ValueError("Cordinate vectors must be real numbers.")
		self.Xdata = np.array([self._x,self._y])
		sz = [max([1,1][i],[len(self._y),len(self._x)][i]) for i in range(2)]

		if len(E) < 1:
			raise ValueError("Electric field data is empty.")
		self._Ex = []
		self._Ey = []
		self._Ez = []

		if type(E) == tuple:
			if len(E) > 0:
				self.scalar = False
				if self.dimens > 2:
					self._Ex = E[0]
				else:
					self._Ex = np.conj(E[0])

			if len(E) > 1:
				if self.dimens > 2:
					self._Ey = E[1]
				else:
					self._Ey = np.conj(E[1])
			
			if len(E) > 2:
				if self.dimens > 2:
					self._Ez = E[2]
				else:
					self._Ez = np.conj(E[2])
		else:
			self.scalar = True
			if self.dimens > 2:
				self._Ex = E
			else:
				self._Ex = np.conj(E)

		self.Edata = (DataFormat(self._Ex,sz),
					DataFormat(self._Ey,sz),
					DataFormat(self._Ez,sz))

		self._Ex = list_to_array(self.Edata[0])
		self._Ey = list_to_array(self.Edata[1])
		self._Ez = list_to_array(self.Edata[2])



		self._Hx = []
		self._Hy = []
		self._Hz = []

		if type(H) == tuple:
			if len(H) > 0:
				self.scalar = False
				self._Hx = H[0]

			if len(H) > 1:
					self._Hy = H[1]
			
			if len(H) > 2:
				self._Hz = H[2]
		else:
			self.scalar = True
			self._Hx = H

		self.Hdata = (DataFormat(self._Hx,sz),
					DataFormat(self._Hy,sz),
					DataFormat(self._Hz,sz))


		self._Hx = list_to_array(self.Hdata[0])
		self._Hy = list_to_array(self.Hdata[1])
		self._Hz = list_to_array(self.Hdata[2])

		self.salut = 5

	def poynting(self):
		"""
		Returns the Poynting vector component z (transverse power density)
		"""
		if self.hasMagnetic() :
			#print(self._Ex,"Ex")
			return  self.Ex*np.conjugate(self.Hy) - self.Ey*np.conjugate(self.Hx)
		else:

			return self.Ex*np.conjugate(self.Ex)

	def power(self):
		"""
		Return power carried by the field in W or W/μm
		"""
		if self.dimens == 3:
			return np.trapz(np.trapz(self._y, self.poynting()),self._x)
		else:
			if self.dimens == 1:
				#print(self.poynting(),self._x)
				return np.trapz(self.poynting(),self._x)
			else:
				return np.trapz(self.poynting(),self._y)


	def normalize(self,P = 1):
		"""
		Normalize the field relatively to it's power.

		P - Normalized Value
		"""
		P0 = self.power()
		for j in range(len(self.Edata)):
			for i in range(len(self.Edata[j])):
				self.Edata[j][i] = self.Edata[j][i]*np.sqrt(P/P0)
		for j in range(len(self.Hdata)):
			for i in range(len(self.Hdata[j])):
				self.Hdata[j][i] = self.Hdata[j][i]*np.sqrt(P/P0)
		return self

	def hasElectric(self):
		"""
		Look if any electric field is define.
		"""
		if np.any([self.Edata]):
			return True
		else:
			return False

	def hasMagnetic(self):
		"""
		Look if any magnetic field is define.
		"""
		if max([np.any(i) for i in self.Hdata]):
			return True
		else:
			return False


	def getMagnitudeE(self):
		"""
		Return the electric field magnitude.
		"""
		A = []
		for i in range(len(self.Edata[0])):
				A.append(sum([abs(self.Edata[j][i])**2 for j in range(len(self.Edata))])**0.5)
		return A
	
	def getMagnitudeH(self):
		"""
		Return the magnetic field magnitude.
		"""
		A = []
		for i in range(len(self.Hdata[0])):
				A.append(sum([abs(self.Hdata[j][i])**2 for j in range(len(self.Hdata))])**0.5)
		return A

	def isScalar(self):
		"""
		Look if there is only one component field.
		"""
		return self.scalar

	def hasX(self):
		"""
		Look if there is any field dendity over X.
		"""
		return (self.dimens == 1) or (self.dimens == 3)

	def hasY(self):
		"""
		Look if there is any field density over Y.
		"""
		return (self.dimens == 2) or (self.dimens == 3)

	def isBidimensional(self):
		"""
		Look if there are X and Y field density.
		"""
		return self.dimens >2

	def isElectroMagnetic(self):
		"""
		Look if there is electric and magnetic field.
		"""
		return self.hasElectric() and self.hasMagnetic()

	def getSize(self):
		"""
		Return the field size.
		"""
		return [max([1,1][i],[len(self._y),len(self._x)][i]) for i in range(2)]

	def offsetCoordinates(self,dx = 0,dy = 0):
		if len(self.Xdata[0]) != 0:
			self.Xdata[0] = [i+dx for i in self.Xdata[0]]
		if len(self.Xdata[0]) != 0:
			self.Xdata[1] = [i+dy for i in self.Xdata[1]]
		return self

	def __str__(self):
		if self.hasX():
			x = f"[{self.Xdata[0][0]},...,{self.Xdata[0][-1]}]"
		else:
			x = "None"
		if self.hasY():
			y = f"[{self.Xdata[1][0]},...,{self.Xdata[1][-1]}]"
		else:
			y = "None"

		return tabulate([['X', x, "X values"], ['Y', y, "Y values"], 
					['E', self.E.shape, "Electrical field shape"],['H', self.H.shape, "Magnetic field shape"]], headers=['parameters', 'Value', 'definition'])




	### Define getter and setter for the Field object ###

	@property
	def x(self):
		return self.Xdata[0]
	
	@property
	def y(self):
		return self.Xdata[1]
	
	@property
	def E(self):
		if self.isScalar:
			return self.Edata[0]
		else:
			return self.Edata

	@property
	def Ex(self):
		return self.Edata[0]

	@property
	def Ey(self):
		return self.Edata[1]

	@property
	def Ez(self):
		return self.Edata[2]

	@property
	def H(self):
		if self.isScalar:
			return self.Hdata[0]
		else:
			return self.Hdata

	@property
	def Hx(self):
		return self.Hdata[0]

	@property
	def Hy(self):
		return self.Hdata[1]

	@property
	def Hz(self):
		return self.Hdata[2]
