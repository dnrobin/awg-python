from core import *
import numpy as np

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

				if len(X) == 2 :
					y = X[1]
					self.dimens = 3
				if len(x) == 0:
					self.dimens = 2
			### To complete later
			else:
				try:
					n,m = np.shape(X)
				except ValueError:
					n = np.shape(X)
					m = 0
				print(n)
				if (n != 0) and (m != 0):
					raise ValueError("Wrong coordinate format. Must be a 1-D vector.")
				x = X
			###
		if len(x) == 0 and len(y) == 0:
			raise Error("At least one coordinate vector must be provided.")


A = Field([0,1,2,3,4,5],[0,1,2],[0,-1,-2])