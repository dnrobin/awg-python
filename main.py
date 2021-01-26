from awg import *
#from awg.Simulate import Simulate
from awg.core import *

import matplotlib.pyplot as plt


F = Field(list_to_array([-2,-1,0,1,2]),list_to_array([0,1,2,1,0])).normalize()


model = AWG(clad = SiO2, core = Si, subs = SiO2, lambda_c = 1.550,
			Ni = 1, No = 9, w = 0.450, h = 0.220, N = 40, m = 75, R = 130,
			d = 2.5, g = 0.4, do = 1.8, wi = 1.5, wo = 1.5, L0 = 20)

options = SimulationOptions()
options.PhaseErrorVariance = 0
options.ModeType = "gaussian"
options.PropagationLoss = 1
options.InsertionLoss = 0.5
Simulate(model,model.lambda_c,Options = options)

results = Spectrum(model,1.55,0.01, Options = options, Samples = 100)

measurements = Analyse(results)