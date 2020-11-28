from warnings import *
import numpy as np

def polyval(a,b=0):
	x = 0
	for i in range(len(a)):
		x += a[i]*b**(len(a)-(i+1))
	return x

def Air(x,T=295):
	if (type(x) == int) or (type(x) == float) or ('numpy.float64' in str(type(x))):
		x = [x]
	return np.ones(len(x))

def Si(x,T = 295):
	if (type(x) == int) or (type(x) == float) or ('numpy.float64' in str(type(x))):
		x = [x]
	"""% Material Sellmeier equation for: Si @ T[20K, 300K], lambda[1.1µm, 5.6µm]
		https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070021411.pdf"""
	n = np.zeros(len(x))
	for i in range(len(x)):
		if (T < 20) or (T > 300):
			warn("Extrapollating Sellmeier equation for Si beyond temperature range of 20K - 300K")
		if (x[i] < 1.1) or (x[i] > 5.6):
			warn("Extrapollating Sellmeier equation for Si beyond range of 1.1µm - 5.6µm")
		S1 = polyval([3.4469e-12,-5.823e-9,4.2169e-6,-0.00020802,10.491],T)
		S2 = polyval([-1.3509e-6,0.0010594,-0.27872,29.166,-1346.6],T)
		S3 = polyval([103.24,678.41,-76158,-1.7621e06,4.4283e07],T)
		x1 = polyval([2.3248e-14, -2.5105e-10, 1.6713e-07, -1.1423e-05, 0.29971],T)
		x2 = polyval([-1.1321e-06, 0.001175, -0.35796, 42.389, -3517.1],T)
		x3 = polyval([23.577, -39.37, -6907.4, -1.4498e05, 1.714e06],T)
		n[i] = (1+(S1*x[i]**2)/(x[i]**2-x1**2)+(S2*x[i]**2)/(x[i]**2-x2**2)+(S3*x[i]**2)/(x[i]**2-x3**2))**0.5
	return n

def Si3N4(x,T = 295):
	"""% Material Sellmeier equation for: Si @ T[20K, 300K], lambda[1.1µm, 5.6µm]
	https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070021411.pdf"""

	if (type(x) == int) or (type(x) == float) or ('numpy.float64' in str(type(x))):
		x = [x]
	
	if (x[0] < 0.31) or (x[-1] > 5.504):
		warn("Extrapollating Sellmeier equation for Si3N4 beyond range of 0.31µm - 5.504µm")
	
	n = np.zeros(len(x))
	
	for i in range(len(x)):
		n[i] = (1+(3.0249/(1-(0.1353406/x[i])**2))+(40314/(1-(1239.842/x[i])**2)))**0.5
	
	return n

def SiO2(x,T = 295):
	"""% Material Sellmeier equation for: Si @ T[20K, 300K], lambda[1.1µm, 5.6µm]
	https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070021411.pdf"""

	if (type(x) == int) or (type(x) == float) or ('numpy.float64' in str(type(x))):
		x = [x]

	if (x[i] < 0.21) or (x[i] > 6.7):
		warn("Extrapollating Sellmeier equation for SiO2 beyond range of 0.21µm - 6.7µm")
	
	n = np.zeros(len(x))
	
	for i in range(len(x)):
		n[i] = (1+(0.6961663/(1-(0.0684043/x[i])**2))+(0.4079426/(1-(0.1162414/x[i])**2))+(0.8974794/(1-(9.8961610/x[i])**2)))**0.5
	return n

