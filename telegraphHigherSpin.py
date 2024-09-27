### Jonathan Curtis
### 09/27/2024
### This is a program to model classical dynamics of a spin S telegraph process

import numpy as np
from matplotlib import pyplot as plt 
from scipy import optimize as opt


##################################################
### Basic functions for transition rate matrix ###
##################################################

### Square lattice form factor function
### Built from scratch to be vectorized
def gk(k):

	### k is assumed to be an N x two-dimensional vector 
	### We want to output an N element array
	return  0.5*(np.cos(k[:,0]) + np.cos(k[:,1])) 

### This now returns Bogoliubov coefficients u,v for a given squeezing parameter
def uk(eta):
	return np.cosh(eta)

def vk(eta):
	return np.sinh(eta)

### We will evaluate this using a Monte Carlo sampling method for the momenta
def gen_momentum_points(npts):

	return 2.*np.pi*np.random.ranf((npts,2)) - np.pi*np.ones((npts,2)) ### This is an array of random momenta in the BZ


#########################################################################
### Integrals which only depend on the Bogoliubov ansatz coefficients ###
#########################################################################

### Now, given a list of momentum points we generate a set of the following integrals 
### It is assumed that for every k point we have an ansatz parameter eta_ks 

### This integral is int_k v_k^2 
def X_int(ks,eta_ks):
	### This integral is int_k v_k^2 
	vks = np.vectorize(vk)(eta_ks)
	return np.mean(vks**2)

### This integral is int_k gamma_k u_k v_k
def M0_int(ks,eta_ks):
	uks = np.vectorize(uk)(eta_ks)
	vks = np.vectorize(vk)(eta_ks)
	gks = gk(ks)

	return np.mean(gks*uks*vks)

### This integral is a function of k and is int_p gamma_{p-k} u_p v_p 
def Mk_int(ks,eta_ks):
	npts = ks.shape[0]
	integrals = np.zeros(npts)
	### For now we will do this the dumb way, and flag for optimization later

	ups = np.vectorize(uk)(eta_ks)
	vps = np.vectorize(vk)(eta_ks)

	for i in range(npts):
		gpks = gk(ks - ks[i])

		integrals[i] = np.mean(gpks*ups*vps)

	return integrals

### This integral is the int_k M_k u_k v_k
def L_int(ks,eta_ks):
	Mks = Mk_int(ks,eta_ks)
	uks = np.vectorize(uk)(eta_ks)
	vks = np.vectorize(vk)(eta_ks)

	return np.mean(Mks*uks*vks)


########################################################################################
### This function will accept a set of ansatz parameters and compute energy function ###
########################################################################################

### The energy functional is quite complicated in general 
### We split it into terms to make it easier and also to examine each piece separately

### This is the contribution due to quadratic terms
### It should be multiplied by J to give the contribution to total energy 
def energy_0(ks,theta,psi_ks,eta_ks):
	M0 = M0_int(ks,eta_ks)
	X = X_int(ks,eta_ks)

	uks = np.vectorize(uk)(eta_ks)
	vks = np.vectorize(vk)(eta_ks)
	gks = gk(ks)

	norm = np.sqrt(np.mean( np.abs(psi_ks)**2 ))

	term1 =  4.* (M0 - X) ### There is a contribution from the sum_k |psi_k|^2 term which ends up contributing sin^2 theta 4(M0 - X) to this 
	term2 = 4.*np.sin(theta)*np.cos(theta)*np.mean( np.real(psi_ks/norm)*( (uks**2 + vk2**2)*gks - 2.*uks*vks )) ### We normalize by the Chevy wavefunction when it appears
	term3 = 4.*np.sin(theta)**2 * np.mean( np.abs(psi_ks/norm)**2 * ( 2.*uks*vks*gks - (uks**2 + vk2**2) ) ) ### This is remaining after pulling out the term which contributes to term 1 

	return term1 + term2 + term3 


### This is the correlation energy due to Hubbard U 
### It should be multiplied by U to give contribution to total energy
### U is a fictitous parameter reflecting the soft-core constraint and should be large but generic (and result should be mostly independent of U ideally)
def energy_U(ks,theta,psi_ks,eta_ks):
	X = X_int(ks,eta_ks)

	uks = np.vectorize(uk)(eta_ks)
	vks = np.vectorize(vk)(eta_ks)
	gks = gk(ks)

	norm = np.sqrt(np.mean( np.abs(psi_ks)**2 ))

	term1 =  X**2 ### There is a contribution from the sum_k |psi_k|^2 term which ends up contributing sin^2 theta X**2 to this 
	term2 = 2.*np.sin(theta)*np.cos(theta)*X*np.mean( np.real(psi_ks/norm)*(2.*uks*vks))
	term3 = 2.*X*np.sin(theta)**2 * np.mean( np.abs(psi_ks/norm)**2 * (uks**2 + vks**2) ) ### This is diagonal terms in the excited state wavefunctions
	term4 = 2.*np.sin(theta)**2 * ( np.abs(np.mean( np.real(psi_ks/norm)*uks*vks )) )**2 ### This is off-diagonal terms in the excited state wavefunctions

	return term1 + term2 + term3 + term4

### This is the correlation energy due to nearest-neighbor attraction J arising from Heisenberg ZZ term 
### It should be multiplied by J to give contribution to total energy
def energy_ZZ(ks,theta,psi_ks,eta_ks):
	return 0. ### We will come back to this later


### This is the total energy as a function of the variational parameters and the Hamiltonian parameters 
def total_energy(J,U,ks,theta,psi_ks,eta_ks):
	return J*energy_0(ks,theta,psi_ks,eta_ks) + U*energy_U(ks,theta,psi_ks,eta_ks) + J*energy_ZZ(ks,theta,psi_ks,eta_ks)


###############################################
### Variational minimization functions here ###
###############################################

### For a fixed set of Bogoliubov coefficients this will find the optimal Chevy parameters
def find_GS_fixed_etas(J,U,ks,eta_ks):

	### We will define here the variational parameters as X = [theta, real(psi_ks), imag(psi_ks)]
	npts = ks.shape[0]

	variational_function = lambda x : total_energy(J,U,ks,x[0],x[1:(npts + 1)] + 1.j*x[(npts+1):],eta_ks) 

	### Now we use, e.g. basin hopping to find the optimal patameters 
	#Initial guess is theta = 0, and chevy parameters real and -1 for all k 
	Xinit = np.zeros(2*npts + 1)
	Xinit[1:(npts+1)] = -1.

	sol = opt.basin_hoppin(variational_function,Xinit)

	energy = sol.fun 
	X_opt = sol.x 

	theta_opt = X_opt[0]
	psi_ks_opt = X_opt[1:(npts+1)] + 1.j*X_opt[(npts+1):]

	return energy, theta_opt, psi_ks_opt


def main():
	

if __name__ == "__main__":
	main()

























