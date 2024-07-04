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
		for k in range(L*L):
			x = k//L
			y = k % L
			tmp[x,y] =  - J*dt*(
				np.sin( thetas[x,y] - thetas[(x+1)//L,y]) 
				+np.sin( thetas[x,y] - thetas[x-1,y]) 
				+np.sin( thetas[x,y] - thetas[x,(y+1)//L]) 
				+np.sin( thetas[x,y] - thetas[x,y-1])
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

	for k in range(L*L):
		x = k//L
		y = k % L
		vorticity[x,y] = ( np.mod(np.thetas[(x+1)//L,y] - thetas[x,y],2.*np.pi) 
			+np.mod( thetas[(x+1)//L,(y+1)//L] - thetas[(x+1)//L,y],2.*np.pi) 
			+np.mod( thetas[x,(y+1)//L] - thetas[(x+1)//L,(y+1)//L],2.*np.pi) 
			+np.mod( thetas[x,y] - thetas[x,(y+1)//L],2.*np.pi) )

	return vorticity

### Computes instantaneous magnetic field noise at NV location given an instantenous profile for the theta field
def calc_NV_noise(z,thetas):

	### First we extract the vorticity field 
	vorticity = np.zeros_like(thetas)

	for k in range(L*L):
		x = k//L
		y = k % L
		vorticity[x,y] = ( np.mod(np.thetas[(x+1)//L,y] - thetas[x,y],2.*np.pi) 
			+np.mod( thetas[(x+1)//L,(y+1)//L] - thetas[(x+1)//L,y],2.*np.pi) 
			+np.mod( thetas[x,(y+1)//L] - thetas[(x+1)//L,(y+1)//L],2.*np.pi) 
			+np.mod( thetas[x,y] - thetas[x,(y+1)//L],2.*np.pi) )


	### Now that we have vorticity we take FFT 
	fft_vort = np.fft.fft2(vorticity)

	### Now we apply the NV filter function 
	nv_filter_func = np.zeros_like(fft_vort)
	qs = np.fft.fft2freqs(vorticity)

	for qx in range(L):
		for qy in range(L):
			nv_filter_func[x,y] = 0.5 * np.exp(-z*np.sqrt(qs[x,y][0]**2 + qs[x,y][1]**2))/np.sqrt(qs[x,y][0]**2 + qs[x,y][1]**2 + 0.0001*qs[0,0][0]**2)

	return np.sum(nv_filter_func*fft_vort)



def main():

	save_data = False
	data_directory = "../../data/"


	L = 40 ### Lattice size -- LxL lattice

	ntemps = 4
	temps = np.linspace(0.1,4.,ntemps)

	nburn = 10000### Time steps we burn initially to equilibrate

	ti = time.time()

	ops = np.zeros(ntemps,dtype=complex)
	vorticity = np.zeros((ntemps,L,L))

	for i in range(ntemps):
		thetas = gen_burned_thetas(L,temps[i],nburn)

		ops[i] = calc_OP(thetas)
		vorticity[i,:,:] = calc_vort(thetas)


	tf = time.time()
	print("Elapsed total time: ",tf-ti,"s")

	for i in range(ntemps):
		plt.imshow(vorticity[i,:,:])
		plt.colorbar()
		plt.show()

	if save_data:
		np.save(data_directory+"thetas.npy",thetas)

	plt.plot(temps,np.abs(ops))
	plt.show()



if __name__ == "__main__":
	main()
