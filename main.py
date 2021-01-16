from awg import *
from awg.core import *
from awg.material import *
from awg.material.Material import Material
import matplotlib.pyplot as plt

# Here we test the framework for now
y = Material(Si3N4)


z = y.dispersion(1.3,1.8)
x = Waveguide(core = Si3N4,subs = SiO2, clad = SiO2,w = 1)
#print(type(x), type(y))
y = x.dispersion(1.3,1.8, point = 500)

print(y)

plt.plot(y[0][0],y[1])
plt.show()

"""x = [[1,2,3],[4,5],[6,7,8,9]]

z = np.zeros(len(x), dtype = int)

for i in range(len(x)):
	z[i] = int(len(x[i]))

y = np.zeros(max(z), dtype = list)
for i in range(len(y)):
	y[i] = []

for i in range(len(x)):
	while len(x[i]) < max(z):
		x[i].append(0)
	for j in range(max(z)):
		y[j].append(x[i][j])

print(x)
print(y)"""
#for i in range()
