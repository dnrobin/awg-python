from awg import *
#from awg.Simulate import Simulate
from awg.core import *

import matplotlib.pyplot as plt

model = AWG(clad = SiO2, core = Si, subs = SiO2, lambda_c = 1.550,
            Ni = 1, No = 9, w = 0.450, h = 0.220, N = 40, m = 75, R = 130,
            d = 2.5, g = 0.4, do = 1.8, wi = 1.5, wo = 1.5, L0 = 20)
options = SimulationOptions()
options.PhaseErrorVariance = 0.12
options.ModeType = "gaussian"
options.PropagationLoss = 1
options.InsertionLoss = 0.5

x= np.linspace(-model.wi*5,model.wi*5,500)
a = 0.6
b = a-0.08
u = np.exp(-((x-b)/a)**2)+np.exp(-((x+b)/a)**2)
F = Field(x,u).normalize()


options.CustomInputField = F


plotfield(iw(model,1.55,u =F))
results_2 = Spectrum(model,1.55,0.01,Options = options, Samples = 100)
plt.plot(results_2.wavelength,10*np.log10(results_2.transmission))
plt.ylim(-40,0)
plt.xlabel("Wavelength ($\mu$m)")
plt.ylabel("Average Transmission (dB)")
plt.show()


"""model = AWG(clad = SiO2, core = Si, subs = SiO2, lambda_c = 1.550,
            Ni = 1, No = 9, w = 0.450, h = 0.220, N = 40, m = 75, R = 130,
            d = 2.5, g = 0.4, do = 1.8, wi = 1.5, wo = 1.5, L0 = 20)

options = SimulationOptions()
options.PhaseErrorVariance = 0.12
options.ModeType = "gaussian"
options.PropagationLoss = 1
options.InsertionLoss = 0.5





x= np.linspace(-model.wi*5,model.wi*5,500)
a = 0.6
b = a-0.08
u = np.exp(-((x-b)/a)**2)+np.exp(-((x+b)/a)**2)
F = Field(x,u).normalize()

#plotfield(F)
options.CustomInputField = F
results = Spectrum(model,1.55,0.01,Options = options, Samples = 100)
plt.plot(results.wavelength,10*np.log10(results.transmission))
plt.ylim(-40,0)
plt.xlabel("Wavelength ($\mu$m)")
plt.ylabel("Average Transmission (dB)")
plt.show()"""

"""data = np.zeros((2,7), dtype = complex)
for n in range(2):
    results = Spectrum(model,1.55,0.01,Options = options,Samples = 100)
    measurements = Analyse(results)
    
    data[n,:] = measurements.Value

fig,ax = plt.subplots()
plt.rcParams["figure.figsize"]=25,20
plt.rcParams.update({'font.size': 15})
ax.boxplot(data,labels = ["Insertion loss","Loss non-uniformity","Channels spacing", "3dB bandwidth","10dB bandwidth", "Adjacent crosstalk","Non-adjacent crosstalk"],showfliers = False)
plt.show()"""

