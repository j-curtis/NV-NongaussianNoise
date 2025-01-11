### Computes noise integral for the model A dynamics 
### Jonathan Curtis 
### 01/11/25

import numpy as np
from scipy import integrate as intg
import time
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors as mclr


### Plotting settings 
#plt.rc('figure', dpi=100)
#plt.rc('figure',figsize=(4,1.7))
#plt.rc('font', family = 'Times New Roman')
#plt.rc('font', size = 14)
#plt.rc('text', usetex=True)
#plt.rc('xtick', labelsize=14)
#plt.rc('ytick', labelsize=14)
#plt.rc('axes', labelsize=18)
#plt.rc('lines', linewidth=2.5)

def sinc(x):
	if x == 0.:
		return 1.

	else:
		return np.sin(x)/x


### This has a maximum at 2q - q^2 = 0 -> q = 2 
### The integral over q is normalized to int_0^infty dq q^2 e^{-q} = 2 
def nv_filter(q):
	return q**2 *np.exp(-np.abs(q)) 


### This is the integrand eveluated in the saddle-point approximation
### we replace the integrals over the three momenta q1,2,3 with delta functions in the magnitude with |qj| = 2 = |sum_j q_j| .
### It is still a function of three frequencies 
def integrand_saddle(w1,w2,w3,x,s):
	ws = [w1,w2,w3]
	W = sum(ws)
	prod = (1. + 4./x**2)/( (W/s)**2 + (1.+4./x**2)**2 )*sinc(W/2.)*1./3.*( np.cos(ws[0]+ws[1]) + np.cos(ws[0]+ws[2]) + np.cos(ws[1]+ws[2]))

	for j in range(3):
		prod *= sinc(ws[j]/2.)/( (ws[j]/s)**2 + (1 + 4/x**2)**2 )

	### This is the volume and normalation factor coming from replacing the momentum integrals with delta functions
	qvol = 1./(2.*np.pi)**6 * (2**4)*(2.*np.pi)**2 ### Check integrals 

	return prod * qvol

### Now we integrate over the three frequencies
### We integrate up to frequencies +- wmax 
def noise_integral(x,s):

	wmax = 30.
	return intg.tplquad(integrand_saddle,-wmax,wmax, -wmax,wmax,-wmax,wmax,args=(x,s))[0]/(np.pi)**3


def main():
	xmin = 0.1
	xmax = 10.
	nx = 4
	xs = np.linspace(xmin,xmax,nx)

	smin = 0.1
	smax = 10.
	ns = 3
	ss = np.linspace(smin,smax,ns)
	
	noises = np.zeros((nx,ns))
	t0 = time.time()
	for i in range(nx):
		x = xs[i]
		for j in range(ns):
			s = ss[j]

			noises[i,j] = noise_integral(x,s)

	t1 = time.time()
	print(t1-t0,"s")
	plt.imshow(noise,origin = 'lower',cmap = 'coolwarm')
	plt.colorbar()
	plt.show()

if __name__ == "__main__":
	main()



