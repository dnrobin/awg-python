from awg import *
from awg.core import *
from awg.material import *
from awg.material.Material import Material
import matplotlib.pyplot as plt
import time
# Here we test the framework for now
"""y = Material(Si3N4)


z = y.dispersion(1.3,1.8)
x = Waveguide(core = Si3N4,subs = SiO2, clad = SiO2,w = 1)
#print(type(x), type(y))
y = x.dispersion(1.3,1.8, point = 500)

print(y)

plt.plot(y[0][0],y[1])
plt.show()"""

#for i in range()
x = AWG(lambda_c = 1.5, clad = SiO2,core =Si,subs = SiO2, Ni = 10)
#F = Field([-4,-3,-2,-1,0,1,2,3,4],[0,1,2,3,4,3,2,1,0])
x.getInputAperture()
t = fpr2(x,1.5, F0 = Field([-4,-3,-2,-1,0,1,2,3,4],[0,1,2,3,4,3,2,1,0],[0,1,2,3,4,3,2,1,0]))

