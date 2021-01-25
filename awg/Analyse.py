from tabulate import tabulate
import numpy as np
from .Simulate import Simulate

def Analyse(results):
	
	lmbda = results.wavelength
	T = results.transmission
	TdB = 10*np.log10(T)
	num_channels = np.shape(T)[1]
	center_channel = int(np.floor(num_channels/2))

	# Insertion loss
	
	IL = abs(max(TdB[:,center_channel]))
	
	
	
	
	
	# 10dB bandwidth
	t0 = TdB[:,center_channel] - IL

	ic = np.argwhere(t0 == max(t0))[0][0]

	ia10 = np.argwhere(t0[0:ic] < -10)[-1][0]

	ib10 = ic + np.argwhere(t0[ic:] < -10)[1][0]

	BW10 = (lmbda[ib10] - lmbda[ia10]) * 1e3
    
    # 3dB bandwidth
	ia3 = np.argwhere(t0[0:ic] < -3)[-1][0]
	ib3 = ic + np.argwhere(t0[ic:] < -3)[1][0]

	BW3 = (lmbda[ib3] - lmbda[ia3]) * 1e3
	
	# 1dB bandwidth
	ia1 = np.argwhere(t0[0:ic] < -1)[-1][0]

	ib1 = ic + np.argwhere(t0[ic:] < -1)[1][0]

	BW1 = (lmbda[ib1] - lmbda[ia1]) * 1e3

	Nu = 0
	Cs = 0
	XT = 0
	XTn = 0

	if num_channels > 1:
		# Non-uniformity
		NU = abs(max(TdB[:,0])) - IL

		# Adjacent crosstalk
		if num_channels < 3:
			if center_channel-1 > 0:
				XT = max(TdB[ia3:ib3,center_channel-1])
			else:
				XT = max(TdB[ia3:ib3,center_channel+1])
		else:
			xt1 = max(TdB[ia3:ib3,center_channel-1])
			xt2 = max(TdB[ia3:ib3,center_channel+1])
			XT = max(xt1,xt2)
	XT -= IL

	# Crosstalk
	XTn = -100
	for i in range(num_channels):
		if (i != center_channel):# and (i != center_channel-1) and (i != center_channel+1) :
			xt = max(TdB[ia3:ib3+1,i])
			XT = max(XT,xt)

	XTn -= IL
	

	# Channel spacing
	if num_channels < 3:
		if center_channel-1 >0:
			ia = np.argwhere(TdB[:,center_channel-1] == max(TdB[:,center_channel-1]))
			CS = 1e3*abs(lmbda[ia]-lmbda[ic])
		else:
			ia = np.argwhere(TdB[:,center_channel+1] == max(TdB[:,center_channel+1]))
			CS = abs(lmbda[ia]-lmbda[ic])*1e3
	else:
		ia = np.argwhere(TdB[:,center_channel-1] == max(TdB[:,center_channel-1]))
		ib = np.argwhere(TdB[:,center_channel+1] == max(TdB[:,center_channel+1]))

		sp1 = abs(lmbda[ia]-lmbda[ic])
		sp2 = abs(lmbda[ib]-lmbda[ic])
		CS = max(sp1,sp2)*1e3

	print(tabulate([['Insertion loss [dB]', IL], ['Loss non-uniformity [dB]', NU], 
					["Channel spacing [nm]", CS],["1dB bandwidth [nm]",BW1],["3dB bandwidth [nm]",BW3],
					["10dB bandwidth [nm]",BW10],["Non-adjacent crosstalk level [dB]",XT],
					["Adjacent crosstalk level [dB]",XT]], headers=['', 'Value']))