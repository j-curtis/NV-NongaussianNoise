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
"""
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


"""


#####################################
### Two-time correlation function ###
#####################################
### This function will find the spin-spin correlation function 
### For the times we pass two-times and return g2(t2,t1) 
def calc_g2(S,t1,t2):
	M = int( 2*S + 1)

	S_vec = np.arange(M) - S*np.ones(M) ### This should be a vector with components [-S,-S+1,...,S-1,S]

	matrix = rate_matrix(S)

	kern = linalg.expm((t2-t1)*matrix)

	return S_vec@kern@S_vec/float(M) ### This is the two-time spin correlation function


######################################
### Four-time correlation function ###
######################################

### This function will find the spin-spin four-time correlation function 
### For the times we pass four times return the function g(t4,t3,t2,t1) 
### The time steps are assumed to be ordered such that t4,t3 > t2,t1
def calc_g4(S,t1,t2,t3,t4):
	M = int( 2*S + 1)

	S_vec = np.arange(M) - S*np.ones(M) ### This should be a vector with components [-S,-S+1,...,S-1,S]
	S_mat = np.diag(S_vec)


	rates = rate_matrix(S)

	### Kernels for the three different time-delays between the four times 

	kern1 = linalg.expm(np.abs(t2-t1)*rates)
	kern2 = linalg.expm( (min(t4,t3) - max(t2,t1))*rates )
	kern3 = linalg.expm(np.abs(t4-t3)*rates)

	
	return S_vec@kern3@kern2@S_mat@kern1@S_vec/float(M) ### This is the four-time spin correlation function


####################################
### Four-time connected cumulant ###
####################################

### This function will find the spin-spin four-time correlation function 
### For the times we pass four times return the function g(t4,t3,t2,t1) 
### The time steps are assumed to be ordered such that t4,t3 > t2,t1
### This is g4(t1,t2,t3,t4) - g2(t1,t2)g2(t3,t4) - g2(t1,t3)g2(t2,t4) - g2(t1,t4)g2(t2,t3)
def calc_g4conn(S,t1,t2,t3,t4):
	g4 = calc_g4(S,t1,t2,t3,t4)

	g2_12 = calc_g2(S,t1,t2)
	g2_13 = calc_g2(S,t1,t3)
	g2_14 = calc_g2(S,t1,t4)

	g2_23 = calc_g2(S,t2,t3)
	g2_24 = calc_g2(S,t2,t4)

	g2_34 = calc_g2(S,t3,t4)

	return - ( g4 - (g2_12*g2_34 + g2_13*g2_24 + g2_14*g2_23) )

####################################
### Fourth order noise Xcumulant ###
####################################
### This function will find the relevant integral for the integral of the connected cumulant over the echo times
### We evaluate using a monte carlo method for the times 
### This relies on int_{-T}^0 dt f(t) = T x sum_j f(t_j) where t_j are randomly chosen from [-T,0] and same for [0,T] integral 
### This then gives T^4 x sum_{jklm} ... for the four randomly sampled times 

def calc_Xcumulant(S,T,npts):
	### T is the echo time and S the spin. npts is the number of monte carlo samples 
	tpts = T*np.random.ranf((npts,4)) ### Random time points all in the interval of [0,T]
	### We how shift the first two to be in the range [-T,0]
	tpts[:,:2] += -np.ones((npts,2))*T 

	### Now we evaluate the cummulant function for each of these time points and tally up the average 
	connected_correlations = np.zeros(npts)

	for i in range(npts):
		connected_correlations[i] = calc_g4conn(S,tpts[i,0],tpts[i,1],tpts[i,2],tpts[i,3])

	return T**4 * np.mean(connected_correlations)




def main():

	S = 3.
	T = 1.
	npts = 5000
	xcum = calc_Xcumulant(S,T,npts)
	print(xcum)
	quit()
	
	S = [ .5,1.,1.5,2.,2.5,3.]
	dt = 0.05
	nt = 100
	times = np.linspace(0.,dt*(nt-1),nt)

	g2s = [ calc_g2(s,dt,nt) for s in S ]

	for s in range(len(S)):
		plt.plot(times,g2s[s])

	plt.plot(times,np.exp(-2.*times),marker='x')
	plt.yscale('log')
	plt.show()





if __name__ == "__main__":
	main()

























