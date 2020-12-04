from core import *
import numpy as np


def DataFormat(D,sz):
	
	if len(D) == 0:
		D = np.zeros(sz)
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

#x = DataFormat(np.array([1,2,3]),[1,3])
#print(x)
class Field:
	def __init__(self,X, E, H = []):
		self.scalar = True
		if len(X) < 1:
			raise ValueError("At least one coordinate vector must be provided.")

		self.y = []
		self.dimens = 1
		
		#if len(X) <= 2:
			### No equivalent of matlab cell in python
			### Use numpy array for now, but will cause error
			### if X have 2 or less coordinate
		if type(X) == tuple:
			self.x = X[0]

			if len(X) > 1 :
				self.y = X[1]
				self.dimens = 3
				if len(self.x) == 0:
					self.dimens = 2
				if len(self.y) == 0:
					self.dimens = 1

		else:
			if len([i for i in np.shape(X)]) > 1:
				raise ValueError("Wrong coordinate format. Must be a 1-D vector.")
			self.x = X

		if (len(self.x) == 0) and (len(self.y) == 0):
			raise Error("At least one coordinate vector must be provided.")
		
		if False in np.isreal(self.x) or False in np.isreal(self.y):
			raise ValueError("Cordinate vectors must be real numbers.")
		self.Xdata = np.array([self.x,self.y])
		sz = max([1,1],[len(self.y),len(self.x)])

		if len(E) < 1:
			raise ValueError("Electric field data is empty.")
		self.Ex = []
		self.Ey = []
		self.Ez = []

		if type(E) == tuple:
			if len(E) > 0:
				self.scalar = False
				self.Ex = E[0]

			if len(E) > 1:
					self.Ey = E[1]
			
			if len(E) > 2:
				self.Ez = E[2]
		else:
			self.scalar = True
			self.Ex = E

		self.Edata = (DataFormat(self.Ex,sz),
					DataFormat(self.Ey,sz),
					DataFormat(self.Ez,sz))

		self.Ex = self.Edata[0]
		self.Ey = self.Edata[1]
		self.Ez = self.Edata[2]



		self.Hx = []
		self.Hy = []
		self.Hz = []

		if type(H) == tuple:
			if len(H) > 0:
				self.scalar = False
				self.Hx = H[0]

			if len(H) > 1:
					self.Hy = H[1]
			
			if len(H) > 2:
				self.Hz = H[2]
		else:
			self.scalar = True
			self.Hx = H

		self.Hdata = (DataFormat(self.Hx,sz),
					DataFormat(self.Hy,sz),
					DataFormat(self.Hz,sz))


		self.Hx = self.Hdata[0]
		self.Hy = self.Hdata[1]
		self.Hz = self.Hdata[2]

		
		

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



	def poynting(self):
		if self.hasMagnetic :
			return  self.Ex*np.conjugate(self.Hy) - self.Ey*np.conjugate(self.Hx)

A = Field([1,2,3],[0,1,2j],[0,-1,-2])
B = A.poynting()
