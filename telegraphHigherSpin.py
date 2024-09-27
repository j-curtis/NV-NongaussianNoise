### Jonathan Curtis
### 09/27/2024
### This is a program to model classical dynamics of a spin S telegraph process

import numpy as np
from matplotlib import pyplot as plt 
from scipy import linalg as linalg


##################################################
### Basic functions for transition rate matrix ###
##################################################

### This generates the gamma matrices for the larger spin S 
### Rate is noramlized so that time is in units of gamma_0^(-1)
def gamma_matrix(S):

	M = int(2*S + 1)
	matrix = np.zeros((M,M))

	for i in range(M):
		m = -S + i
		for j in range(M):
			n = -S + j
			if i == j + 1 or i == j -1:
				matrix[i,j] = S*(S+1) - m*n

	return matrix 
 

### This generates the rate matrices for the larger spin S 
### Rate is noramlized so that time is in units of gamma_0^(-1)
### We have rate[i,j] =  rate j -> i and therefore this involves 
### rate[i,j] = gamma[i,j] - sum_{k != i} gamma[k,i] delta_{i,j}
def rate_matrix(S):

	M = int( 2*S + 1)
	matrix = gamma_matrix(S)

	matrix -= np.sum(matrix,axis=0)*np.eye(M)

	return matrix 


#################################################
### Obtaining the propagator for time step dt ###
#################################################

### This function will find the propagator for transitioning from state m to state n in time t and return the matrix
### This will be done by exponentiating the matrix for a short time-step
### Long time evolution can be implemented by repeated matrix multiplication with the time step
def kern_step(S,dt):
	M = int( 2*S + 1)
	matrix = rate_matrix(S)

	kern = linalg.expm(dt*matrix)

	return kern


#####################################
### Two-time correlation function ###
#####################################

### This function will find the spin-spin correlation function 
### For the times we pass dt and a number of steps nt and return values for all integers up to nt 
def calc_g2(S,dt,nt):
	M = int( 2*S + 1)

	S_vec = np.arange(M) - S*np.ones(M) ### This should be a vector with components [-S,-S+1,...,S-1,S]

	matrix = rate_matrix(S)

	dkern = linalg.expm(dt*matrix)
	kern = np.eye(M)

	g2s = np.zeros(nt)
	g2s[0] = S_vec@kern@S_vec/float(M) ### This is the two-time spin correlation function

	for j in range(1,nt):
		kern = dkern@kern
		g2s[j] = S_vec@kern@S_vec/float(M) ### This is the two-time spin correlation function

	return g2s


def main():
	
	S = 1.5
	dt = 0.05
	nt = 30
	times = np.linspace(0.,dt*(nt-1),nt)

	g2 = calc_g2(S,dt,nt)

	plt.plot(g2)
	plt.show()





if __name__ == "__main__":
	main()

























