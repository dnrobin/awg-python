from awg.core import *
import matplotlib.pyplot as plt
import numpy as np
from . import *





def plotfield(X, Y = [], **kwargs):
	_in = kwargs.keys()

	if "PlotPhase" in _in:
		PlotPhase = kwargs["PlotPhase"]
	else:
		PlotPhase = False

	if "PlotPower" in _in:
		PlotPower = kwargs["PlotPower"]
	else:
		PlotPower = False

	if "UnwrapPhase" in _in:
		UnwrapPhase = kwargs["UnwrapPhase"]
	else:
		UnwrapPhase = False

	if  "NormalizePhase" in _in:
		NormalizePhase = kwargs["NormalizePhase"]
	else:
		NormalizePhase = False

	if "Figure" in _in:
		Figure = kwargs["Figure"]
	Figure = False



	if	str(type(X)) == "<class 'awg.Field.Field'>":
		F = X # Ignore Y
	else:
		F = Field(X,Y)
	
	rows = 1

	if F.isElectroMagnetic():
		rows = 2

	if PlotPower:
		rows += 1

	if Figure != False:
		fig = Figure
	else:
		fig = plt.figure()

	def plotField1D(x,u,xname,uname,subplot_position):
		ax1 = fig.add_subplot(subplot_position,)
		ax2 = ax1.twinx()
		ax1.set_xlabel(f"{xname}($\mu$m)")
		if PlotPhase:
			u1 = np.abs(u)**2
			u2 = np.angle(u)
			if UnwrapPhase:
				u2 = np.unwrap(u2)
			if NormalizePhase:
				u2 /= np.pi
			u1label = f"|{uname}|$^2$"
			u2label = f"$\phi$({uname})"
		else:
			u1 = np.real(u)
			u2 = np.imag(u)
			u1label = f"Re({uname})"
			u2label = f"Im({uname})"
		if PlotPhase and NormalizePhase:
			meany = np.mean(u2)
			miny = meany + min(-1,min(u2)-meany)
			maxy = meany + max(1,max(u2)-meany)
			ax2.set_ylim(miny,maxy)
			u2label = "$\\frac{\phi}{\pi}$"+f"({uname})"
		
		ax1.plot(x,u1,color = "b")
		ax1.set_ylabel(u1label)
		ax2.plot(x,u2, color = "r")
		ax2.set_ylabel(u2label)

	def plotField2D(x,y,u,xname,yname,uname):
		ax1 = fig.add_subplot(211)
		ax2 = fig.add_subplot(212)
		if PlotPhase:
			u1 = np.abs(u)**2
			u2 = np.angle(u)
			if UnwrapPhase:
				u2 = unwrap(u2)
			if NormalizePhase:
				u2 /= np.pi
			utitle = f"|{uname}|$^2$"
		else:
			u1 = np.real(u)
			u2 = np.imag(u)
			u1title = f"Re({uname})"
		cmap = plt.get_cmap("jet")
		im = ax1.pcolormesh(x,y,u1,cmap = cmap)
		fig.colorbar(im,ax = ax1)
		cf = ax2.contourf(x,y,u2, cmap = cmap)
		fig.colorbar(cg,ax = ax2)
		ax1.set_title(utitle)
		ax1.set_xlabel(f"{xname}($\mu$m)")
		ax1.set_ylabel(f"{yname}($\mu$m)")

	if F.isBidimensional():
		### TO DO when Field will accept 2D Field
		pass
	else:
		a = F.x
		t = "x"
		if F.hasY():
			a = F.y
			t = "y"
		if F.isScalar():
			if F.hasElectric():
				plotField1D(a,F.E,t,"E",rows*100+10+1)
			if F.hasMagnetic():
				i = 0
				if F.hasElectric():
					i = 1
				plotField1D(a,F.H,t,"H",rows*100+10+i+i)
		else:
			if F.hasElectric():
				r = rows*100+30
				plotField1D(a,F.Ex,t,"E$_x$",r+1)
				plotField1D(a,F.Ey,t,"E$_y$",r+2)
				plotField1D(a,F.Ez,t,"E$_z$",r+3)
			if F.hasMagnetic():
				i = 0
				if F.hasElectric():
					i = 3
				plotField1D(a,F.Hx,t,"H$_x$",r+1 + i)
				plotField1D(a,F.Hy,t,"H$_y$",r+2 + i)
				plotField1D(a,F.Hx,t,"H$_x$",r+3 + i)

	plt.tight_layout()














