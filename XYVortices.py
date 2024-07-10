### Code to compute stochastic dynamics of XY model and tabulate higher order statistics for vortex non-gaussian noise 
### Jonathan Curtis
### 06/26/2024

import numpy as np
from matplotlib import pyplot as plt 
import time 

#######################################
### THESE METHODS ARE FOR RUNNING SIMULATION IN PYTHON
#######################################

### Function which will accept simulation parameters and return a thermalized initial state
def gen_burned_thetas(L,T,nburn,J=1.,dt = .05):
	"""Generates a sequence of angles for a given system size, temperature, number of time steps, Josephson coupling (default is 1.) and time step size (default is 5% J)"""
	### Uses periodic boundary conditions
	### defaul time step is 5% of J

	thetas = np.zeros((L,L))
	tmp = np.zeros_like(thetas)

	for nt in range(1,nburn):
		for x in range(L):
			for y in range(L):
				tmp[x,y] =  - J*dt*(
					np.sin( thetas[x,y] - thetas[(x+1)//L,y] ) 
					+np.sin( thetas[x,y] - thetas[x-1,y] ) 
					+np.sin( thetas[x,y] - thetas[x,(y+1)//L] ) 
					+np.sin( thetas[x,y] - thetas[x,y-1] )
					)
		
		thetas += tmp
		thetas +=  np.random.normal(0.,2.*T*dt,size=(L,L))

	return thetas

### Given a time-slice value of thetas[x,y] this returns the spatial average of the order parameter 
def calc_OP(thetas):
	"""Calculates the spatial average of order parameter over space"""
	OP = np.mean(np.exp(1.j*thetas),axis=(-1,-2))

	return OP

### Given a time-slice of values of thetas[x,y] this returns the vorticity density 
def calc_vort(thetas):
	vorticity = np.zeros_like(thetas)

	L = thetas.shape[0]

	for x in range(L):
		for y in range(L):
			vorticity[x,y] = ( np.fmod(thetas[(x+1)//L,y] - thetas[x,y],2.*np.pi) 
				+np.fmod(thetas[(x+1)//L,(y+1)//L] - thetas[(x+1)//L,y],2.*np.pi) 
				+np.fmod(thetas[x,(y+1)//L] - thetas[(x+1)//L,(y+1)//L],2.*np.pi) 
				+np.fmod(thetas[x,y] - thetas[x,(y+1)//L],2.*np.pi) )

	return vorticity


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
### vorticity is assumed to be a single time slice and is LxL
def fft_filtered(vorticity,z_list):

	### First we extract how many z points we will be computing for 
	num_zs = len(z_list)

	### We will need FFT only once for all z points
	### We compute here 
	### We compute the real fft but insist it have the same output shape as the input array, for convenience
	fft = np.fft.rfft2(data,s=vorticity.shape)

	### We also need the corresponding momentum space points to compute the filters 
	### This also is needed only once for all z points
	qs = np.linspace(0.,2.*np.pi,L)

	### Now we compute the filter functions, which has one for each z point we want 
	filter_funcs = np.zeros((num_zs,L,L))

	### Output array for filtered FFT at each distance z in the list 
	out = np.zeros((num_zs,L,L),dtype=complex)

	for nz in range(num_zs):
		z = z_list[nz]

		for nx in range(L):
			for ny in range(L):
				q = np.sqrt(qs[nx]**2 + qs[ny]**2)
			
				filter_funcs[nz,nx,ny] = np.exp(-2.*z*q)/(2.*q + .00001)**2 ### For q -> 0 we shift slightly, it should be ultimately suppressed anyways

		out[nz,:,:] = fft*filter_funcs[nz,:,:]


	return out


def main():

	z_list = np.array([1.,3.,10.,30.,100.,300.,1000.,3000.])

	data_directory = "../data/"
	files = ["vorticity_L=30_t=5000_T=1.500000"]

	paths = [data_directory + file for file in files]

	data_files = [ load_csv(path) for path in paths]

	for data in data_files:

		for nt in range(len(data.shape[0])):
			bfield_z = fft_filtered(data[nt,:,:],z_list)

if __name__ == "__main__":
	main()












