from .core import *
import numpy as np


def DataFormat(D,sz):
	if len(D) == 0:
		D = np.zeros(sz)[0]
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
				self._Ex = E[0]

			if len(E) > 1:
					self._Ey = E[1]
			
			if len(E) > 2:
				self._Ez = E[2]
		else:
			self.scalar = True
			self._Ex = E

		self.Edata = (DataFormat(self._Ex,sz),
					DataFormat(self._Ey,sz),
					DataFormat(self._Ez,sz))

		self._Ex = self.Edata[0]
		self._Ey = self.Edata[1]
		self._Ez = self.Edata[2]



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


		self._Hx = self.Hdata[0]
		self._Hy = self.Hdata[1]
		self._Hz = self.Hdata[2]

		

	def poynting(self):
		if self.HasMagnetic() :
			return  self._Ex*np.conjugate(self._Hy) - self._Ey*np.conjugate(self._Hx)
		else:
			return self._Ex*np.conj(self._Ex)

	def power(self):
		if self.dimens == 3:
			return np.trapz(np.trapz(self._y, self.poynting()),self._x)
		else:
			if self.dimens == 1:

				return np.trapz(self.poynting(),self._x)
			else:
				return np.trapz(self.poynting(),self._y)

	def normalize(self,P = 1):
		P0 = abs(self.power())
		for i in range(len(self.Edata)):
			for j in range(len(self.Edata[0])):
				self.Edata[i][j] = self.Edata[i][j]*(P/P0)**0.5
		for i in range(len(self.Hdata)):
			for j in range(len(self.Hdata[0])):
				self.Hdata[i][j] = self.Hdata[i][j]*(P/P0)**0.5
		return self

	def hasElectric(self):
		if np.any([self.Edata]):
			return True
		else:
			return False

	def hasMagnetic(self):
		if np.any([self.Hdata]):
			return True
		else:
			return False


	def getMagnitudeE(self):
		A = []
		for i in range(len(self.Edata[0])):
				A.append(sum([abs(self.Edata[j][i])**2 for j in range(len(self.Edata))])**0.5)
		return A
	
	def getMagnitudeH(self):
		A = []
		for i in range(len(self.Hdata[0])):
				A.append(sum([abs(self.Hdata[j][i])**2 for j in range(len(self.Hdata))])**0.5)
		return A

	def isScalar(self):
		return self.scalar

	def hasX(self):
		return (self.dimens == 1) or (self.dimens == 3)

	def hasY(self):
		return (self.dimens == 2) or (self.dimens == 3)

	def isBidimensional(self):
		return self.dimens >2

	def isElectroMagnetic(self):
		return self.HasElectric and self.HasMagnetic

	def getSize(self):
		return [max([1,1][i],[len(self._y),len(self._x)][i]) for i in range(2)]

	def offsetCoordinates(self,dx = 0,dy = 0):
		if len(self.Xdata[0]) != 0:
			self.Xdata[0] = [i+dx for i in self.Xdata[0]]
		if len(self.Xdata[0]) != 0:
			self.Xdata[1] = [i+dy for i in self.Xdata[1]]
		return self

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
