from awg import *
#from awg.Simulate import Simulate
from awg.core import *

import matplotlib.pyplot as plt


model = AWG(clad = SiO2, core = Si, subs = SiO2, lambda_c = 1.550,
            Ni = 1, No = 9, w = 0.450, h = 0.220, N = 40, m = 75, R = 130,
            d = 2.5, g = 0.4, do = 1.8, wi = 1.5, wo = 1.5, L0 = 20)
F1 = iw(model,model.lambda_c+0.0031,0, ModeType ="solve",points = 100)
lmbda = model.lambda_c + 0.0031
modetype = "solve"
F2 = fpr1(model,lmbda,F1,points = 500)
F3 = aw(model,lmbda,F2,ModeType = modetype)
F4 = fpr2(model,lmbda,F3, points = 500)
T = ow(model,lmbda,F4,ModeType = modetype)


plt.bar([i+1 for i in range(len(T))],T, color = "b")
plt.show()

