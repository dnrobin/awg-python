from tabulate import tabulate
import numpy as np
from .Simulate import Simulate

class Analyse:
	def __init__(self,results):
	
		lmbda = results.wavelength
		T = results.transmission
		TdB = 10*np.log10(T)
		num_channels = np.shape(T)[1]
		center_channel = int(np.floor(num_channels/2))

		# Insertion loss
		
		self.IL = abs(max(TdB[:,center_channel]))
		
		#print(TdB,num_channels,center_channel)
		
		
		
		# 10dB bandwidth
		t0 = TdB[:,center_channel] - self.IL

		ic = np.argwhere(t0 == max(t0))[0][0]

		ia10 = np.argwhere(t0[0:ic+1] < -10)[-1][0]

		ib10 = ic + np.argwhere(t0[ic:] < -10)[1][0]

		self.BW10 = (lmbda[ib10] - lmbda[ia10]) * 1e3
	    
	    # 3dB bandwidth
		ia3 = np.argwhere(t0[0:ic+1] < -3)[-1][0]
		ib3 = ic + np.argwhere(t0[ic:] < -3)[1][0]

		self.BW3 = (lmbda[ib3] - lmbda[ia3]) * 1e3
		

		self.NU = 0
		self.CS = 0
		self.XT = 0
		self.XTn = 0

		if num_channels > 1:
			# Non-uniformity
			self.NU = abs(max(TdB[:,0])) - self.IL

			# Adjacent crosstalk
			if num_channels < 3:
				if center_channel-1 > 0:
					self.XT = max(TdB[ia3:ib3,center_channel-1])
				else:
					self.XT = max(TdB[ia3:ib3,center_channel+1])
			else:
				xt1 = max(TdB[ia3:ib3,center_channel-1])
				xt2 = max(TdB[ia3:ib3,center_channel+1])
				self.XT = max(xt1,xt2)
		self.XT -= self.IL
		#print(ia3,ib3)
		# Crosstalk
		self.XTn = -100
		for i in range(num_channels):
			if (i != center_channel):# and (i != center_channel-1) and (i != center_channel+1) :
				xt = max(TdB[ia3:ib3+1,i])
				self.XTn = max(self.XTn,xt)


		self.XTn -= self.IL
		

		# Channel spacing
		if num_channels < 3:
			if center_channel-1 >0:
				ia = np.argwhere(TdB[:,center_channel-1] == max(TdB[:,center_channel-1]))
				self.CS = 1e3*abs(lmbda[ia]-lmbda[ic])
			else:
				ia = np.argwhere(TdB[:,center_channel+1] == max(TdB[:,center_channel+1]))
				self.CS = abs(lmbda[ia]-lmbda[ic])*1e3
		else:
			ia = np.argwhere(TdB[:,center_channel-1] == max(TdB[:,center_channel-1]))
			ib = np.argwhere(TdB[:,center_channel+1] == max(TdB[:,center_channel+1]))

			sp1 = abs(lmbda[ia]-lmbda[ic])
			sp2 = abs(lmbda[ib]-lmbda[ic])
			self.CS = max(sp1,sp2)*1e3
		self.Value = [self.IL,self.NU,self.CS,self.BW3,self.BW10,self.XT,self.XTn]
	def __str__(self):
		return tabulate([['Insertion loss [dB]', self.IL], ['Loss non-uniformity [dB]', self.NU], 
					["Channel spacing [nm]", self.CS],["3dB bandwidth [nm]",self.BW3],
					["10dB bandwidth [nm]",self.BW10],["Adjacent crosstalk level [dB]",self.XT],
					["Non-adjacent crosstalk level [dB]",self.XTn]], headers=['', 'Value'])