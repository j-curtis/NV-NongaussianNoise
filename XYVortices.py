### Code to compute stochastic dynamics of XY model and tabulate higher order statistics for vortex non-gaussian noise 
### Jonathan Curtis
### 06/26/2024

import numpy as np
from matplotlib import pyplot as plt 
import time 



def genThetas(L,T,ntimes,J=1.,dt = .05):
	"""Generates a sequence of angles for a given system size, temperature, number of time steps, Josephson coupling (default is 1.) and time step size (default is 5% J)"""
	### Uses periodic boundary conditions
	### defaul time step is 5% of J

	thetas = np.zeros((ntimes,L,L))

	for nt in range(1,ntimes):
		for k in range(L*L):
			x = k//L
			y = k % L
			thetas[nt,x,y] = thetas[nt-1,x,y] - J*dt*(
				np.sin( thetas[nt-1,x,y] - thetas[nt-1,(x+1)//L,y]) 
				+np.sin( thetas[nt-1,x,y] - thetas[nt-1,x-1,y]) 
				+np.sin( thetas[nt-1,x,y] - thetas[nt-1,x,(y+1)//L]) 
				+np.sin( thetas[nt-1,x,y] - thetas[nt-1,x,y-1])
				)

		thetas[nt,:,:] +=  np.random.normal(0.,2.*T*dt,size=(L,L))

	return thetas

def calcOP(thetas):
	"""Calculates the spatial average of order parameter over space"""
	s = thetas.shape
	ntimes = s[0]
	OP = np.zeros(ntimes,dtype=complex)
	OP = np.mean(np.exp(1.j*thetas),axis=(-1,-2))

	return OP


def main():

	L = 50### Lattice size -- LxL lattice
	T = .7*.89 ### Literature says BKT transition is at approximately .89 J 

	nburn = 5000### Time steps we burn initially to equilibrate
	ntimes = 3000### Number of times steps we calculate and measure for

	ti = time.time()

	thetas = genThetas(L,T,nburn+ntimes)

	tf = time.time()
	print("Elapsed total time: ",tf-ti,"s")

	np.save("thetas.npy",thetas)



if __name__ == "__main__":
	main()

