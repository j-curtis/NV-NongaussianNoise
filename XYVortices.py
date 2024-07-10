### Code to compute stochastic dynamics of XY model and tabulate higher order statistics for vortex non-gaussian noise 
### Jonathan Curtis
### 06/26/2024

import numpy as np
from matplotlib import pyplot as plt 
import time 


data_directory = "../../data/"

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



### HERE IS A SET OF FUNCTIONS FOR PROCESSING VORTICITY PROFILES FROM C++ SIMULATIONS

### This method returns a numpy data file obtained from reading a csv file of the format of LxL doubles per line with each line a time step 
def load_csv(fname):

	data = np.genfromtxt(fname+".csv",delimiter=" ")

	nt = data.shape[0]
	L = int(np.sqrt(data.shape[1]))

	data = data.reshape(nt,L,L)

	return data


### This method will perform an fft on the vorticity data 
### vorticity is assumed to be a single time slice and is LxL
def fft_vorticity(vorticity):
	
	fft_out = np.fft.fft2(data)
	return fft_out

	
def main():




if __name__ == "__main__":
	main()












