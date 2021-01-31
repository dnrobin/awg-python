from awg import *
#from awg.Simulate import Simulate
from awg.core import *

import matplotlib.pyplot as plt


model = AWG(clad = SiO2, core = Si, subs = SiO2, lambda_c = 1.550,
            Ni = 1, No = 9, w = 0.450, h = 0.220, N = 40, m = 75, R = 130,
            d = 2.5, g = 0.4, do = 1.8, wi = 1.5, wo = 1.5, L0 = 20)
F1 = iw(model,model.lambda_c+0.0031,0, ModeType ="gaussian",points = 100)
lmbda = model.lambda_c + 0.0031
modetype = "solve"
F2 = fpr1(model,lmbda,F1,points = 500)
#print(F2.Ex)
plotfield(F2,PlotPhase = True,UnwrapPhase = True, NormalizePhase = True)
#plt.show()