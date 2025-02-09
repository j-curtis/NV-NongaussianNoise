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

@np.vectorize
def sinc(x):
	if x == 0.:
		return 1.

	else:
		return np.sin(x)/x


### Evaluates the full integral without saddle-point 
### This invovles nine integration variables and is quite hefty...
def noise_integral_full(x,s,Nq,Nf):
	### First we need to draw a sequence of momentum samples
	### These are drawn according to importance sampling of the near-field distribution q e^(-q) where q is a two-dimensional integral 
	### This is a Gamma distribution with variance theta = 1 and parameter k = 2 

	rng = np.random.default_rng()

	normalization_q = (4.*np.pi)**3 ### the normalization of the momentum sampling is fixed to match the analytic value of the integral 

	qs = rng.gamma(3,1.,(Nq,3)) ### This is a set of N samples of 3 momenta with magnitudes drawn from Gamma distribution with theta =1  
	angles = 2.*np.pi*rng.random((Nq,3)) ### Three random angles 

	Qs = np.sqrt(  np.sum(qs * np.cos(angles) ,axis=-1)**2 + np.sum(qs * np.sin(angles) ,axis=-1)**2 ) ### This should give us the vector sum of the sampled momenta 

	f1s = Qs*np.exp(-Qs)*(1. + Qs**2/x**2) ### These will be a contribution to the integrand which does not depend on frequencies 

	widths = s * ( np.ones((Nq,3,Nf)) + np.multiply.outer(qs**2/x**2,np.ones(Nf) )) ### This will be the width of the Lorentzians used to sample the frequency integrals 

	normalization_f = ( (np.pi*s**2)**3)/np.prod(widths[...,0],axis=-1) ### This should be the correct normalization for converting the monte carlo sampled integral into the analyic one
	### Note this depends on the momentum sample

	integrand = normalization_q*f1s * normalization_f 

	### Now we sample the frequencies according to a Lorentzian 
	### For each momentum sample we will sample Nf frequencies as these are correlated to the momentum values 
	ws = rng.standard_cauchy((Nq,3,Nf))
	ws *= widths

	Ws = np.sum(ws,axis=1)

	sincs = sinc(Ws/2.)*np.prod(sinc(ws/2.),axis=1) ### Product of all the sinc factors in the integral

	cosines = np.zeros((Nq,Nf))

	for i in range(3):
		cosines += 1./3. * np.cos(ws[:,i,:] + ws[:,i-1,:])


	integrand *= np.mean(sincs*cosines/( (Ws/s)**2 + (1.+np.outer(Qs,np.ones(Nf))**2/x**2)**2 ) ,axis=-1)

	return np.mean(integrand) 



### Evaluates the full noise integral without saddle-point in the static limit
### This invovles only momentum integrals 
### Time is measured in units of correlation time tau_c
### Length is measured in units of correlation length xi_c
def noise_integral_static(x,s,a,tdp,N):
	### First we need to draw a sequence of momentum samples
	### These are drawn according to importance sampling of the near-field distribution q e^(-q) where q is a two-dimensional integral 
	### This is a Gamma distribution with variance theta = 1 and parameter k = 2 

	rng = np.random.default_rng()

	normalization = 2.**10/(np.pi**3)*s**4/(x**10)* (1./tdp)**4 *a**2 ### the normalization of the momentum sampling is fixed to match the analytic value of the integral 
	### tdp = a^3 / (gS mu_B^2 mu_0) is the dipolar coupling time, and a is the lattice constant in units of correlation length (should be small) 

	qs = rng.gamma(3,1.,(N,3)) ### This is a set of N samples of 3 momenta with magnitudes drawn from Gamma distribution with theta =1  
	angles = 2.*np.pi*rng.random((N,3)) ### Three random angles 

	Qs = np.sqrt(  np.sum(qs * np.cos(angles) ,axis=-1)**2 + np.sum(qs * np.sin(angles) ,axis=-1)**2 ) ### This should give us the vector sum of the sampled momenta 

	integrand = Qs*np.exp(-Qs) / ( 4. + np.sum(qs**2,axis=-1)/x**2 + Qs**2/x**2 ) * np.prod(1./( 1. + qs**2/x**2),axis=-1) ### These will be a contribution to the integrand which does not depend on frequencies 


	return np.mean(integrand) *normalization



def main():

	dataDirectory = "../data/01312025_1/"

	a = .01
	tdp = .01

	N = int(1e8)

	nx = 50
	xs = np.logspace(-4.,2.,nx)

	ss = np.array([0.01,0.03,0.1])
	ns = len(ss)

	print("N: ",N)
	print("nx: ",nx)
	print("ns: ",ns)
	print("a: ",a)
	print("tdp: ",tdp)

	
	noises = np.zeros((nx,ns))
	times = np.zeros((nx,ns))

	for i in range(nx):
		for j in range(ns):
			t0 = time.time()
			noises[i,j] = noise_integral_static(xs[i],ss[j],a,tdp,N)
			t1 = time.time()
			times[i,j] = t1- t0
			print(t1-t0,"s")



	np.save(dataDirectory+"xs.npy",xs)
	np.save(dataDirectory+"ss.npy",ss)
	np.save(dataDirectory+"times.npy",times)
	np.save(dataDirectory+"noises.npy",noises)



if __name__ == "__main__":
	main()



