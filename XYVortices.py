### Code to compute stochastic dynamics of XY model and tabulate higher order statistics for vortex non-gaussian noise 
### Jonathan Curtis
### 06/26/2024

import numpy as np
from matplotlib import pyplot as plt 
import time 

#######################################
### THESE METHODS ARE FOR RUNNING SIMULATION IN PYTHON
#######################################

### This will be a vectorizeable implementation of the lattice derivative needed often 
### Specifically it will return the sum sum_{k in nn of j} sin(theta_k - theta_j)
def compact_lattice_derivative(thetas):
	dtheta_px = np.roll(thetas,1,axis=0) - thetas
	dtheta_mx = np.roll(thetas,-1,axis=0) - thetas 
	dtheta_py = np.roll(thetas,1,axis=1) - thetas 
	dtheta_my = np.roll(thetas,-1,axis=1) - thetas

	return np.sin(dtheta_px) + np.sin(dtheta_mx) + np.sin(dtheta_py) + np.sin(dtheta_my) 


### Function which will accept simulation parameters and return a thermalized initial state
def gen_burned_thetas(L,T,nburn,J=1.,dt = .1):
	"""Generates a sequence of angles for a given system size, temperature, number of time steps, Josephson coupling (default is 1.) and time step size (default is 5% J)"""
	### Uses periodic boundary conditions

	thetas = np.zeros((L,L))
	for nt in range(1,nburn):
		thetas += J*dt*compact_lattice_derivative(thetas) + np.random.normal(0.,2.*T*dt,size=(L,L))

	return thetas
	

### This function generates a simulation which is a set of time-traces of theta for a given set of parameters
### The returned field theta will be a large array of shape 
### Nsample x Ntimes x L x L 
### The definition of the samples is that each sampled configuration has the same burned in initial condition
def run_sim(L,T,nburn,nsample,ntimes,J=1.,dt=0.1):

	out = np.zeros((nsample,ntimes,L,L))
	for ns in range(nsample):

		### Initialize the dynamics for each sampling to be the same burned in configuration
		out[ns,0,:,:] = gen_burned_thetas(L,T,nburn,J,dt)

		for nt in range(1,ntimes):
			out[ns,nt,:,:] = out[ns,nt-1,:,:] + J*dt*compact_lattice_derivative(out[ns,nt-1,...]) + np.random.normal(0.,2.*T*dt,size=(L,L))

	return out

### Given a time-slice value of thetas[x,y] this returns the spatial average of the order parameter 
def calc_OP(thetas):
	"""Calculates the spatial average of order parameter over space"""
	OP = np.abs(np.mean(np.exp(1.j*thetas),axis=(-1,-2)))

	return OP

### Given a time-slice of values of thetas[x,y] this returns the vorticity density 
def calc_vort(thetas):
	out = np.zeros_like(thetas)

	tmp1 = thetas
	tmp2 = np.roll(tmp1,1,axis=0)
	out += np.fmod(tmp2-tmp1,2.*np.pi) 
	
	tmp1 = tmp2
	tmp2 = np.roll(tmp1,1,axis=1)
	out += np.fmod(tmp2-tmp1,2.*np.pi) 
	
	tmp1 = tmp2 
	tmp2 = np.roll(tmp1,-1,axis=0)
	out += np.fmod(tmp2-tmp1,2.*np.pi)
	
	tmp1 = tmp2
	tmp2 = np.roll(tmp1,-1,axis=1)
	out += np.fmod(tmp2-tmp1,2.*np.pi)

#	L = thetas.shape[0]
#
#	for x in range(L):
#		for y in range(L):
#			vorticity[x,y] = ( np.fmod(thetas[(x+1)%L,y] - thetas[x,y],2.*np.pi) 
#				+np.fmod(thetas[(x+1)%L,(y+1)%L] - thetas[(x+1)%L,y],2.*np.pi) 
#				+np.fmod(thetas[x,(y+1)%L] - thetas[(x+1)%L,(y+1)%L],2.*np.pi) 
#				+np.fmod(thetas[x,y] - thetas[x,(y+1)%L],2.*np.pi) )

	return out


#######################################
### HERE IS A SET OF FUNCTIONS FOR PROCESSING VORTICITY PROFILES FROM C++ SIMULATIONS
#######################################

### This method returns a numpy data file obtained from reading a csv file of the format of LxL doubles per line with each line a time step 
def load_csv(fname):

	data = np.genfromtxt(fname+".csv",delimiter=" ")

	nt = data.shape[0]
	L = int(np.sqrt(data.shape[1]))

	data = data.reshape(nt,L,L)

	return data


### This method will take the spatial vorticity profile and compute the filtered FFT at a depth z (accepts an array of distances z)
### vorticity is assumed to be an NtxLxL array of all time slices 
def fft_filtered(vorticity,z_list):

	### First we extract how many z points we will be computing for 
	num_zs = len(z_list)

	num_ts = vorticity.shape[0]
	L = vorticity.shape[1]

	### We will need FFT only once for all z points
	### We compute here 
	### We compute the real fft but insist it have the same output shape as the input array, for convenience
	ffts = np.fft.fft2(vorticity,axes=(-1,-2))

	### We also need the corresponding momentum space points to compute the filters 
	### This also is needed only once for all z points
	qs = np.linspace(0.,2.*np.pi,L)

	### Now we compute the filter functions, which has one for each z point we want 
	filter_funcs = np.zeros((num_zs,L,L))

	### Output array for filtered FFT at each distance z in the list 
	out = np.zeros((num_ts,num_zs),dtype=complex)

	for nt in range(num_ts):
		for nz in range(num_zs):
			z = z_list[nz]

			for nx in range(L):
				for ny in range(L):
					q = np.sqrt(qs[nx]**2 + qs[ny]**2)
				
					filter_funcs[nz,nx,ny] = np.exp(-2.*z*q)/(2.*q + .00001)**2 ### For q -> 0 we shift slightly, it should be ultimately suppressed anyways

			### We now need to sum over all the momenta to reduce the output down to just the distance-dependent time-dependent field 
			### We use the mean so that we essentially normalize by the number of momentum points -- I think this will correspond to the integral in continuum
			out[nt,nz] = np.mean(ffts[nt,:,:]*filter_funcs[nz,:,:])

	return out


def process_files():

	t0 = time.time()
	z_list = np.array([1.,3.,10.,30.,100.,300.])

	data_directory = "../data/"
	files = ["vorticity_L=30_t=5000_T=1.500000","vorticity_L=50_t=20000_T=1.500000"]

	paths = [data_directory + file for file in files]

	data_files = [ load_csv(path) for path in paths]

	for data in data_files:

		field_vs_t_z = fft_filtered(data,z_list)

		plt.plot(field_vs_t_z[:,0])
		plt.show()
		plt.plot(field_vs_t_z[:,2])
		plt.show()
		plt.plot(field_vs_t_z[:,4])
		plt.show()

	tf = time.time()
	print("Total time: ",tf-t0,"s")


def main():

	L = 50
	T = 0.7
	nburn = 1000
	nsample = 100
	ntimes = 100 
	
	t0 = time.time()

	thetas = run_sim(L,T,nburn,nsample,ntimes)

		
	t1 = time.time()
	print(t1-t0,"s")

if __name__ == "__main__":
	main()












