import types
import numpy as np
from warnings import warn

# here we define some pre-existing material functions

def Air(wvl, T = 295):
	if (type(wvl) == int) or (type(wvl) == float) or ('numpy.float64' in str(type(wvl))):
		wvl = [wvl]
	return np.ones(len(wvl))

def Si(wvl,T = 295):
	""" Material Sellmeier equation for: Si @ T[20K, 300K], lambda[1.1µm, 5.6µm]
	https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070021411.pdf"""

	if (type(wvl) == int) or (type(wvl) == float) or ('numpy.float64' in str(type(wvl))):
		wvl = [wvl]
	
	if (wvl[0] < 1.1) or (wvl[-1] > 5.6):
			warn("Extrapollating Sellmeier equation for Si beyond range of 1.1µm - 5.6µm")
	
	if (T < 20) or (T > 300):
		warn("Extrapollating Sellmeier equation for Si beyond temperature range of 20K - 300K")


	n = np.zeros(len(wvl))
	for i in range(len(wvl)):

		
		S1 = np.polyval([3.4469e-12,-5.823e-9,4.2169e-6,-0.00020802,10.491],T)
		S2 = np.polyval([-1.3509e-6,0.0010594,-0.27872,29.166,-1346.6],T)
		S3 = np.polyval([103.24,678.41,-76158,-1.7621e06,4.4283e07],T)
		x1 = np.polyval([2.3248e-14, -2.5105e-10, 1.6713e-07, -1.1423e-05, 0.29971],T)
		x2 = np.polyval([-1.1321e-06, 0.001175, -0.35796, 42.389, -3517.1],T)
		x3 = np.polyval([23.577, -39.37, -6907.4, -1.4498e05, 1.714e06],T)
		n[i] = (1+(S1*wvl[i]**2)/(wvl[i]**2-x1**2)+(S2*wvl[i]**2)/(wvl[i]**2-x2**2)+(S3*wvl[i]**2)/(wvl[i]**2-x3**2))**0.5

	return n

def SiO2(wvl, T = 295):
	""" Material model (Sellmeier) for: SiO2 @ 20?C over (0.21µm - 6.7µm)
	https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson"""

	if (type(wvl) == int) or (type(wvl) == float) or ('numpy.float64' in str(type(wvl))):
		wvl = [wvl]

	if (wvl[0] < 0.21) or (wvl[-1] > 6.7):
		warn("Extrapollating Sellmeier equation for SiO2 beyond range of 0.21µm - 6.7µm")
	
	n = np.zeros(len(wvl))
	
	for i in range(len(wvl)):
		n[i] = (1+(0.6961663/(1-(0.0684043/wvl[i])**2))+(0.4079426/(1-(0.1162414/wvl[i])**2))+(0.8974794/(1-(9.8961610/wvl[i])**2)))**0.5

	return n


def Si3N4(wvl, T = 295):
	"""% Material Sellmeier equation for: Si @ T[20K, 300K], lambda[1.1µm, 5.6µm]
	https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070021411.pdf"""

	if (type(wvl) == int) or (type(wvl) == float) or ('numpy.float64' in str(type(wvl))):
		wvl = [wvl]
	
	if (wvl[0] < 0.31) or (wvl[-1] > 5.504):
		warn("Extrapollating Sellmeier equation for Si3N4 beyond range of 0.31µm - 5.504µm")
	
	n = np.zeros(len(wvl))
	
	for i in range(len(wvl)):
		n[i] = (1+(3.0249/(1-(0.1353406/wvl[i])**2))+(40314/(1-(1239.842/wvl[i])**2)))**0.5
	
	return n

def Ge(wvl, T = 295):
  
	"""Material model (Sellmeier) for: Ge @ T[20K, 300K], lambda[1.9µm, 5.5µm]
	https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070021411.pdf"""

	if (type(wvl) == int) or (type(wvl) == float) or ('numpy.float64' in str(type(wvl))):
		wvl = [wvl]
	
	if (wvl[0] < 1.9) or (wvl[-1] > 5.5):
			warn("Extrapollating Sellmeier equation for Si beyond range of 1.9µm - 5.5µm")
	
	if (T < 20) or (T > 300):
		warn("Extrapollating Sellmeier equation for Si beyond temperature range of 20K - 300K")

	n = np.zeros(len(wvl))
	
	for i in range(len(wvl)):

		
		S1 = np.polyval([-4.8624e-12, 2.226e-08, -5.022e-06, 0.0025281, 13.972],T)
		S2 = np.polyval([4.1204e-11, -6.0229e-08, 2.1689e-05, -0.003092, 0.4521],T)
		S3 = np.polyval([-7.7345e-06, 0.0029605, -0.23809, -14.284, 751.45],T)
		x1 = np.polyval([5.3742e-12, -2.2792e-10, -5.9345e-07, 0.00020187, 0.38637],T)
		x2 = np.polyval([9.402e-12, 1.1236e-08, -4.9728e-06, 0.0011651, 1.0884],T)
		x3 = np.polyval([-1.9516e-05, 0.0064936, -0.52702, -0.96795, -2893.2],T)
		n[i] = (1+(S1*wvl[i]**2)/(wvl[i]**2-x1**2)+(S2*wvl[i]**2)/(wvl[i]**2-x2**2)+(S3*wvl[i]**2)/(wvl[i]**2-x3**2))**0.5
	
	return n


