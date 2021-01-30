import numpy as np
import matplotlib.pyplot as plt
import math
from core import *

def splitatmedian(x):
	n = len(x)
	i = math.floor((n+1)/2)

	if n+1 % 2 > 0:
		l = x[0:i]
		r = x[i:]
	else:
		l = x[0:i]
		r = x[i:]
	return l,r

def quartiles(x,Limits = 1.5):
	if type(x) == list:
		x = list_to_array(x)
	size = x.shape
	if len(size) >= 2:
		m,n = size
		print(m,n)
		if (m > 1) and (n > 1):
			raise ValueError("Array must be one dimension")
		elif n > 1:
			x = x[0]
		
	d = np.sort(x)
	l,u = splitatmedian(d)

	q1 = np.median(l)
	q3 = np.median(u)
	IQR = q3-q1

	w = [(i < (q1-Limits*IQR)) or (i> (q3+Limits*IQR)) for i in d]

	o = d[w]

	p = np.setdiff1d(d,o)
	q = np.array([p[0],q1,np.median(d),q3,p[-1]])
	return q

print(quartiles(np.array([1,2,3,4,5,6,7,8,9,10,30,35]),1))


def AWG_Boxplot(data, label = []):
	m,n = data.shape
	print(m,n)
	ax = plt.figure()
	def box(max_value,min_value,q1,q3,median_value,center_value,label):
		ax.plot([center_value-0.25,center_value+0.25],[max_value,max_value],color = "k")
		ax.plot([])
		return x
	if m == 7:
		pass
	elif n == 7:
		IL = data[:,0]
		NU = data[:,1]
		CS = data[:,2]
		BW3 = data[:,3]
		BW10 = data[:,4]
		XT = data[:,5]
		XTn = data[:,6]

		print(np.)

	print(XTn)



x = np.ones((5,7))
for i in range(5):
	x[i:,] = [1+i,2+i,3+i,4+i,5+i,6+i,7+2*i]
print(x)
AWG_Boxplot(x)


