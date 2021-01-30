from awg import *
#from awg.Simulate import Simulate
from awg.core import *

import matplotlib.pyplot as plt


model = AWG(clad = SiO2, core = Si, subs = SiO2, lambda_c = 1.550,
			Ni = 1, No = 9, w = 0.450, h = 0.220, N = 40, m = 75, R = 130,
			d = 2.5, g = 0.4, do = 1.8, wi = 1.5, wo = 1.5, L0 = 20)

options = SimulationOptions()
options.PhaseErrorVariance = 0.1
options.ModeType = "gaussian"
options.PropagationLoss = 1
options.InsertionLoss = 0.5
#Simulate(model,model.lambda_c,Options = options)

data = np.zeros((10,7))

for i in range(10):

	results = Spectrum(model,1.55,0.01, Options = options, Samples = 50)

	measurements = Analyse(results)

	data[i,:] = measurements.Value
plt.boxplot((data[:,0],data[:,1],data[:,2],data[:,3],data[:,4],data[:,5],data[:,6]))
plt.legend(["Insertion Loss", "Loss non-uniformity","Channel Spacing","3dB Bandwith","10dB Bandwidth","Adjacent Crosstalk","Crosstalk"])

print(measurements)

plt.show()
