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


### This is the integrand eveluated in the saddle-point approximation and Monte Carlo sampling for the frequencies 
def noise_integral(x,s,N):
	g = s*(1+4./x**2)
	prefactor = 64.*np.pi**5 *(1.+4./x**2)*s**6 / g**3

	rng = np.random.default_rng()

	samples = np.zeros(N) ### This will be a list of samples of the integrand 

	for i in range(N):
		ws = rng.standard_cauchy(3)*g ### Generator N samples of three frequencies, scaled by gamma since this is by default variance unity

		W = sum(ws)

		samples[i] = sinc(ws[0]/2.)*sinc(ws[1]/2.)*sinc(ws[2]/2.)*sinc(W/2.)
		samples[i] *= 1./3.*( np.cos( ws[0] + ws[1] ) + np.cos( ws[0] + ws[2] ) + np.cos( ws[1] + ws[2] ) )
		samples[i] *= 1./( (W/s)**2 + (1. + 4./x**2)**2 )
		samples[i] *= prefactor 

	### From our samples we return the mean and std dev 

	return np.mean(samples)


### This will be an optimized sample processor 
def noise_integral_opt(x,s,N):
	### First we want to generate a set of samples 
	g = s*(1+4./x**2)
	rng = np.random.default_rng()
	ws = rng.standard_cauchy((N,3))*g ### Generator N samples of three frequencies, scaled by gamma since this is by default variance unity
	Ws = np.sum(ws,axis=-1)

	sinc_v = np.vectorize(sinc)

	sincs = np.ones(N)
	cosines = np.zeros(N)


	for i in range(3):
		sincs[:] *= sinc_v(ws[:,i]/2.)
		cosines[:] += 1./3. * ( np.cos(ws[:,i] + ws[:,i-1]) )

	sincs *= sinc_v(Ws/2.)
	lorentzian_Ws = 1./( (Ws[:]/s)**2 + (1.+4./x**2)**2 )


	samples = 64.*np.pi**5 *(1.+4./x**2)*s**6 / g**3*sincs*lorentzian_Ws*cosines

	return np.mean(samples)


def main():
	nx = 20
	xs = np.logspace(-1,2,nx)
	s = 3.
	N = int(1e6)

	noise = np.zeros(nx)
	t0 = time.time()

	for i in range(nx):
		x = xs[i]


		noise[i] = noise_integral_opt(x,s,N)

	t1 = time.time()
	print(t1-t0,"s")
	plt.plot(xs,noise)
	plt.xscale('log')
	#plt.yscale('log')
	plt.show()

if __name__ == "__main__":
	main()



