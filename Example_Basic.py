from awg import *
from awg.core import *
import matplotlib.pyplot as plt
import numpy as np






lmbda = [1.40, 1.46, 1.51, 1.54, 1.55, 1.57, 1.60]
index = [3.74, 3.41, 3.20, 3.09, 3.06, 2.99, 2.91]

myMaterial = Material.Material([lmbda,index])
n = myMaterial.index(1.5)
Ng = myMaterial.groupindex(1.55)
print(Ng)