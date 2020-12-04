from core import *
import numpy as np
print(np.isreal([0,1,2,3,4,5]))
class Field:
	def __init__(self,X, E,H):
		self.scalar = True
		if len(X) < 1:
			raise ValueError("At least one coordinate vector must be provided.")

		y = []
		self.dimens = 1
		
		if len(X) <= 2:
			### No equivalent of matlab cell in python
			### Use numpy array for now, but will cause error
			### if X have 2 or less coordinate
			if type(X) == np.ndarray:
				x = X[0]

				if len(X) > 1 :
					y = X[1]
					self.dimens = 3
					if len(x) == 0:
						self.dimens = 2
			### To complete later
			else:
				try:
					n,m = np.shape(X) # cause error if vector 1 1D or if x and y are not the same length
				except ValueError:
					n = np.shape(X)
					m = 0
				if (n != 0) and (m != 0):
					raise ValueError("Wrong coordinate format. Must be a 1-D vector.")
				
			###
		x = X
		if (len(x) == 0) and (len(y) == 0):
			raise Error("At least one coordinate vector must be provided.")
		if len(y) == 0:
			if False in np.isreal(x):
				raise ValueError("Cordinate vectors must be real numbers.")
		else:
			if False in np.isreal(x) or False in np.isreal(y):
				raise ValueError("Cordinate vectors must be real numbers.")

		self.Xdata = np.array([x,y])
		sz = max([1,1],[len(y),len(x)])

		if len(E) < 1:
			raise ValueError("Electric field data is empty.")
		Ex = []
		Ey = []
		Ez = []

A = Field(np.array([[0,1,2,3,4,5],[0,2,1]]),[0,1,2],[0,-1,-2])
