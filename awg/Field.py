from core import *
import numpy as np


#print([i for i in np.shape(([1,2,3],[1,2,3]))])

def isvector(a):
	dim = 0
	for i in a:
		if len(i) == 0:
			pass
		else:
			dim +=1
	if dim > 1:
		return True
	else:
		return False

print(isvector(([1,2,3],[2,3])))



def DataFormat(D,sz):
	if len(D) == 0:
		D = np.zeros(sz)
		return D
	print(np.shape(D),sz)
	shape_D = [1]
	if len(np.shape(D))> 1:
		shape_D = [i for i in np.shape(D)]
	else:
		shape_D.append(np.shape(D)[0])
	print(shape_D)
	s =[]
	for i in sz:
		if i > 1:
			s.append(True)
		else:
			s.append(False)
	if False not in s:
		if shape_D != sz:
			raise ValueError("Wrong data format. The field must contain the same number of rows as the y-coordinate points and the same number of columns as the x-coordinate points.")
	


	return D

x = DataFormat(([1,2,3],[2,3]),[1,3])
print(x)
class Field:
	def __init__(self,X, E,H):
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

		print(self.Ex)



A = Field(([1,2,3],[5,2]),([0,1,2j],[5,2]),[0,-1,-2])
