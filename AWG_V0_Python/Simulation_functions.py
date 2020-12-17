import types
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy.integrate import trapz
from scipy.optimize import root
from Materials import *
from AWG_function import *
import inspect
from tabulate import tabulate
### La variable AWG sera un dictionnaire




class AWG_obj:
	def __init__(self):
		self.verification()
		
	def verification(self):
		_in = inspect.getmembers(self)[26:-1]
		_in = [_in[i][0] for i in range(len(_in))]
		for i in range(len(_in)):
				if _in[i] == "clad":
					if (type(self.clad) == types.FunctionType) or (type(self.clad) == float) or (type(self.clad) == int):
						self.clad = _in[i]
					else:
						raise ValueError("The cladding must be a material or a float representing its refractive index.")
				elif _in[i] == "core":
					if (type(self.core) == types.FunctionType) or (type(self.core) == float) or (type(self.core) == int):
						self.core = _in[i]
					else:
						raise ValueError("The core must be a material or a float representing its refractive index.")
				elif _in[i] == "subs":
					if (type(self.subs) == types.FunctionType) or (type(self.subs) == float) or (type(self.subs) == int):
						self.subs = _in[i]
					else:
						raise ValueError("The substrate must be a material or a float representing its refractive index.")
				elif _in[i] == "lambda_c":
					if (type(self.lambda_c) == float) or (type(self.lambda_c) == int):
						self.lambda_c = _in[i]
					else:
						raise ValueError("The central wavelength [um] must be a float or an integer.")			
				elif _in[i] == "w" :
					if (type(self.w) == float) or (type(self.w) == int):
						self.w = _in[i]
					else:
						raise ValueError("The array waveguide core width [um] must be a float or an integer.")	
				elif _in[i] == "h" :
					if (type(self.h) == float) or (type(self.h) == int):
						self.h = _in[i]
					else:
						raise ValueError("The array waveguide core height [um] must be a float or an integer.")	
				elif _in[i] == "N":
					if  type(self.N) == int:
						self.N = _in[i]
					else:
						raise ValueError("The number of output must be an integer.")	
				elif _in[i] == "M" :
					if type(self.M) == int:
						self.M = _in[i]
					else:
						raise ValueError("The number of arrayed waveguide must be an integer.")	
				elif _in[i] == "m" :
					if type(self.m) == int:
						self.m = _in[i]
					else:
						raise ValueError("The number of arrayed waveguide must be an integer.")	
				elif _in[i] == "R":
					if (type(self.R) == float) or (type(self.R) == int):
						self.R = _in[i]
					else:
						raise ValueError("The focal length must be a float or an integer.")
				elif _in[i] == "d" :
					if (type(self.d) == float) or (type(self.d) == int):
						self.d = _in[i]
					else:
						raise ValueError("The arrayed waveguide spacing [um] must be float or an integer.")	
				elif _in[i] == "do" :
					if (type(self.do) == float) or (type(self.do) == int):
						self.do = _in[i]
					else:
						raise ValueError("The output waveguide spacing [um] must be float or an integer.")		
				elif _in[i] == "wi":
					if (type(self.wi) == float) or (type(self.wi) == int):
						self.wi = _in[i]
					else:
						raise ValueError("The input aperture width [um] be a float or an integer.")
				elif _in[i] == "wg" :
					if (type(self.wg) == float) or (type(self.wg) == int):
						self.wg = _in[i]
					else:
						raise ValueError("The arrayed aperture width [um] must be float or an integer.")	
				elif _in[i] == "wo" :
					if (type(self.wo) == float) or (type(self.wo) == int):
						self.wo = _in[i]
					else:
						raise ValueError("The output aperture width [um] must be float or an integer.")
				elif _in[i] == "verification":
					pass
				else:
					raise KeyError(f"{i} is not defined by the AWG parameters.")



AWG = AWG_obj()

AWG.clad = SiO2
AWG.core = Si3N4
AWG.subs = SiO2

AWG.lambda_c = 1.550

AWG.w = 2
AWG.h = 0.1

AWG.N = 5
AWG.M = 34
AWG.m = 165
AWG.R = 200
AWG.d = 6
AWG.do = 6
AWG.wi = 6
AWG.wg = 2
AWG.wo = 6


def AWG_simulate(AWG,lmbda0,**kwargs):

	pp = kwargs

	if "plot" not in pp.keys():
		pp["plot"] = False
	if "sample" not in pp.keys():
		pp["sample"] = 250
	if "gaussian" not in pp.keys():
		pp["gaussian"] = False


	if type(AWG.clad) == types.FunctionType:
		nclad = AWG.clad(lmbda0)
	else:
		nclad = AWG.clad
	
	if type(AWG.core) == types.FunctionType:
		ncore = AWG.core(lmbda0)
	else:
		ncore = AWG.core
	
	if type(AWG.subs) == types.FunctionType:
		nsubs = AWG.subs(lmbda0)
	else:
		nsubs = AWG.subs
	nc = eim_index(lmbda0, AWG.w, AWG.h, np.inf, nsubs, ncore, nsubs, N = 1)[0]

	ns = slab_index(lmbda0, AWG.h, nclad, ncore, nsubs, N = 1)

	dl = AWG.m * AWG.lambda_c / eim_index(AWG.lambda_c, AWG.w, AWG.h, np.inf, nsubs, ncore, nsubs, N = 1)[0]
	def aperture_mode(lmbda0,w,h,e,x):
		
		if pp["gaussian"]:
			V = 2*np.pi/lmbda0 * w * np.sqrt(ncore**2-nclad**2)
			w_mode = w*(0.5+1/(V-0.6))
			E = (2/(np.pi*w_mode**2))**0.25*np.exp(-x**2/w_mode**2)
		else:
			_ , y, Ek, _, _ = eim_mode(lmbda0,w,h,e,nclad,ncore,nsubs, x = x, y = np.arange(-h,2*h,0.01))
			for i in range(len(y)):
				if (y[i] < 0.01) and (y[i] > -0.01):
					where_y = i-1
			E = Ek[:,where_y,0]
			#print(E)
			E = E/np.sqrt(trapz(abs(E)**2,x))
		return E
	
	def iw_function(lmbda0):
		s = np.linspace(-0.5, 0.5, pp["sample"],dtype = complex)*(2*AWG.wi)
		E = aperture_mode(lmbda0, AWG.wi, AWG.h, np.inf, s)
		return s,E

	def fpr1_function(lmbda0,s0,E0):
		s = np.linspace(-0.5,0.5,2*pp["sample"], dtype = complex)*(AWG.M+4)*AWG.d
		x = AWG.R*np.sin(np.conjugate(s)/AWG.R)
		z = AWG.R*np.cos(np.conjugate(s)/AWG.R)
		E = rsdi(lmbda0/ns,E0,s0,z,x)[0]
		return s,E

	def aw_function(lmbda0,s0,E0):
		E = np.zeros(len(E0))

		for i in range(0,AWG.M):
			sc = (i  - (AWG.M - 1)/2) * AWG.d
			Em = aperture_mode(lmbda0,AWG.wg,AWG.h,np.inf,s0-sc)
			#print(np.mean(Em),sc)
			P = overlap(s0,E0,Em)
			L = i*dl
			D = np.exp(-1j*2*np.pi/lmbda0*nc*L)
			E = E + np.sqrt(P)*D*Em[:]*rect((s0 - sc)/AWG.d).conj()
			#print(rect((s0 - sc)/AWG.d).conj())
		return s0,E

	def fpr2_function(lmbda0,s0,E0):
		s = np.linspace(-0.5,0.5,pp["sample"],dtype = complex)*(AWG.N + 4)*AWG.do;
		theta = np.conjugate(s0)/AWG.R
		xp = AWG.R*np.tan(theta)
		dp = AWG.R*1/(np.cos(theta)) - AWG.R
		Ep = E0*np.exp(1j*2*np.pi/lmbda0*ns*dp)
		E = rsdi(lmbda0/ns,Ep,xp,[AWG.R],s)[0]
		return s,E

	def ow_function(lmbda0,s0,E0):
		T = np.zeros(AWG.N, dtype = complex)
		for i in range(AWG.N):
			sc = (i - (AWG.N - 1)/2) * AWG.do
			Em = aperture_mode(lmbda0, AWG.wo, AWG.h, np.inf, s0 - sc)
			T[i] = overlap(s0,E0,Em)
			#print(T[i])
		return T



	s0,E0 = iw_function(lmbda0)
	s1,E1 = fpr1_function(lmbda0,s0,E0)
	s2,E2 = aw_function(lmbda0,s1,E1)
	x3,E3 = fpr2_function(lmbda0,s2,E2)
	T = ow_function(lmbda0,x3,E3)
	if pp["plot"]:
		fig,ax = plt.subplots()
		ax.plot(s0,E0, color = "b", linewidth = 2)
		ax.set_xlabel('$x_0$ [$\\mu$m]')
		ax.set_ylabel('|$E_x$($x_0$)|')
		ax.set_title(f"Input Mode Field (\u03BB = {lmbda0*1e3:.0f} nm)")
		ax.set_ylim(min(E0),max(E0))
		ax.set_xlim(min(s0),max(s0))
		
		fig,ax1 = plt.subplots()
		ax1.plot(s1,abs(E1),color = "b", linewidth = 2)
		ax1.set_ylabel("|E$_x$|")
		ax1.set_title(f"FPR1 Diffracted Field ($\\lambda$ = {lmbda0*1e3:.0f}  nm)")
		ax1.set_xlabel("x$_1$ [$\\mu$m]")
		ax1.set_ylim(min(abs(E1)),max(abs(E1)))
		ax1.set_xlim(min(s1),max(s1))
		ax2 = ax1.twinx()
		ax2.set_ylim(min(np.angle(E1)/np.pi),max(np.angle(E1)/np.pi))
		ax2.plot(s1,np.angle(E1)/np.pi, color = "r", linewidth = 2)
		ax2.set_ylabel("$\\phi/\\pi$")
		

		fig,ax3 = plt.subplots()
		ax3.plot(s2,abs(E2), color = "b", linewidth = 2)
		ax3.set_ylabel("|E$_x$|")
		ax3.set_xlabel("x$_2$ [$\\mu$m]")
		ax3.set_title(f"Output Mode Field (\u03BB = {lmbda0*1e3:.0f} nm)")
		ax3.set_ylim(min(abs(E2)),max(abs(E2)))
		ax3.set_xlim(min(s2),max(s2))
		ax4 = ax3.twinx()
		ax4.plot(s2,np.unwrap(np.angle(E2)), color = "r", linewidth = 2)
		ax4.set_ylabel("$\\phi$[rad]")
		ax4.set_ylim(min(np.unwrap(np.angle(E2))),max(np.unwrap(np.angle(E2))))

		fig,ax5 = plt.subplots()
		ax5.plot(x3,abs(E3), color = "b", linewidth = 2)
		ax5.set_ylabel("|E$_x$|")
		ax5.set_xlabel("x$_1$ [$\\mu$m]")
		ax5.set_title(f"FPR2 Diffracted Field ($\\lambda$ = {lmbda0*1e3:.0f}  nm)")
		ax5.set_ylim(min(abs(E3)),max(abs(E3)))
		ax5.set_xlim(min(x3),max(x3))
		ax6 = ax5.twinx()
		ax6.plot(x3, np.angle(E3)/np.pi, color = "r", linewidth = 2)
		ax6.set_ylabel("$\\phi/\\pi$")
		ax6.set_ylim(min(np.angle(E3)/np.pi),max(np.angle(E3)/np.pi))
		
		fig,ax7 = plt.subplots()
		ax7.bar([i for i in range(1,AWG.N+1)],T, color = "b")
		ax7.set_xlim(0,AWG.N+1)
		ax7.set_title(f"Output transmission ($\\lambda$ = {lmbda0*1e3:.0f} nm)")
		ax7.set_ylim(0,1)
		ax7.set_xlabel("Channel #")
		ax7.set_ylabel("Transmission")

	return T

def AWG_spectrum(AWG,lmbda0,bandwidth, **kwargs):
	pp = kwargs
	if "plot" not in pp.keys():
		pp["plot"] = False
	if "sample" not in pp.keys():
		pp["sample"] = 100
	if "gaussian" not in pp.keys():
		pp["gaussian"] = False

	sample_pts = pp["sample"]

	lmbda = lmbda0 + np.linspace(-0.5,0.5,sample_pts)*bandwidth
	T = np.zeros((sample_pts,AWG.N))

	for i in range(sample_pts):
		T[i,:] = abs(AWG_simulate(AWG, lmbda[i], gaussian = pp["gaussian"]))
		print(f"{i+1}/{sample_pts}")
	fig, ax1 = plt.subplots()
	if pp["plot"]:
		for i in range(AWG.N):
			ax1.plot(lmbda, 10*np.log10(T[:,i]), LineWidth = 2, label = f"Out {i+1}")
		ax1.set_xlabel('$\\lambda$ [µm]')
		ax1.set_ylabel('Transmission [dB]')
		ax1.set_title("AWG Transmission Spectrum")
		ax1.set_ylim(-40,0)
		ax1.set_xlim(min(lmbda),max(lmbda))
		plt.legend()
		#plt.show()
	return T, lmbda

def AWG_analyse(lmbda,T):
	TdB = 10*np.log10(T)
	num_channels = np.shape(T)[1]
	center_channel = int(np.floor(num_channels/2))

	# Insertion loss
	
	IL = abs(max(TdB[:,center_channel]))
	
	# Non-uniformity
	
	NU = abs(max(TdB[:,0])) - IL
	
	# 10dB bandwidth
	t0 = TdB[:,center_channel]

	ic = np.argwhere(t0 == max(t0))[0][0]

	ia10 = np.argwhere(t0[0:ic] < -10)[-1][0]

	ib10 = ic + np.argwhere(t0[ic:] < -10)[1][0]

	BW10 = (lmbda[ib10] - lmbda[ia10]) * 1e3
    
    # 3dB bandwidth
	ia3 = np.argwhere(t0[0:ic] < -3)[-1][0]

	ib3 = ic + np.argwhere(t0[ic:] < -3)[1][0]

	BW3 = (lmbda[ib3] - lmbda[ia3]) * 1e3
	

	
	# Crosstalk level
	XT = -100
	for i in range(num_channels):
		if (i != center_channel) and (i != center_channel-1) and (i != center_channel+1) :
			xt = max(TdB[ia3:ib3+1,i])
			XT = max(XT,xt)


	# Adjacent crosstalk level

	AT = -100
	for i in [center_channel-1, center_channel+1]:
		at = max(TdB[ia3:ib3+1,i])
		AT = max(AT,at)

	# 1dB bandwidth

	ia1 = np.argwhere(t0[0:ic] < -1)[-1][0]

	ib1 = ic + np.argwhere(t0[ic:] < -1)[1][0]

	BW1 = (lmbda[ib1] - lmbda[ia1]) * 1e3
   	
   	# Channel spacing

	ia_s = np.argwhere(TdB[:, center_channel - 1] == max(TdB[:, center_channel - 1]))[-1][0]
	ib_s = np.argwhere(TdB[:, center_channel + 1] == max(TdB[:, center_channel + 1]))[-1][0]

	sp1 = abs(lmbda[ia_s] - lmbda[ic])
	sp2 = abs(lmbda[ib_s] - lmbda[ic])
	CS = max(sp1, sp2) * 1e3;

	print(tabulate([['Insertion loss [dB]', IL], ['Loss non-uniformity [dB]', NU], ["Channel spacing [nm]", CS],["1dB bandwidth [nm]",BW1],["3dB bandwidth [nm]",BW3],["10dB bandwidth [nm]",BW10],["Non-adjacent crosstalk level [dB]",XT],["Adjacent crosstalk level [dB]",AT]], headers=['', 'Value']))
#print(overlap([-1,-0.5,0,0.5,1],[-4.72e-03-3.11e-03j, -4.78e-03-3.15e-03j, -4.84e-03-3.19e-03j,-4.90e-03-3.22e-03j, -4.96e-03-3.26e-03j],[-310000.72e-03-52.11e-03j, -310000.78e-03-52.15e-03j, -310000.84e-03-52.19e-03j,-310000.90e-03-52.22e-03j, -310000.96e-03-52.26e-03j]))
#eim_mode(1.550,AWG.w,AWG.h,np.inf,SiO2,Si3N4,SiO2)
AWG_simulate(AWG,AWG.lambda_c, plot = True)
#T , lmbda = AWG_spectrum(AWG,AWG.lambda_c,0.012,sample = 100, plot = True, gaussian = True)
#AWG_analyse(lmbda,T)

plt.show()
