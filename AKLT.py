### Few site AKLT Loschmidt Echo
### Jonathan Curtis 
### 12/21/24

import numpy as np
import time as time 
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors as mclr
import scipy as scp


### We first construct single-site spin-1 representation
S_site = [ 1./np.sqrt(2.)*np.array([[0.j,1.,0.],[1.,0.,1.],[0.,1.,0.]]), 1./np.sqrt(2.)*np.array([[0.,-1.j,0.],[1.j,0.,-1.j],[0.,1.j,0.]]), np.array([[1.,0.,0.j],[0.,0.,0.],[0.,0.,-1.]]) ]

 
L = 6

### Manually construct operators by brute force ... I am not proud of this 
### Now we form tensor product 
### We will start with L = 6 sites which is ~ 800 dimensional hilbert space 
one = np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3),np.eye(3) ) ) ) ) )  

Sx = [ np.kron( S_site[0], np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3),np.eye(3) ) ) ) ) ) ,
np.kron( np.eye(3), np.kron( S_site[0], np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3),np.eye(3) ) ) ) ) ),
np.kron( np.eye(3), np.kron( np.eye(3), np.kron( S_site[0], np.kron( np.eye(3), np.kron( np.eye(3),np.eye(3) ) ) ) ) ),
np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3), np.kron( S_site[0], np.kron( np.eye(3),np.eye(3) ) ) ) ) ),
np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3), np.kron( S_site[0],np.eye(3) ) ) ) ) ),
np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3),S_site[0] ) ) ) ) ) ]

Sy = [ np.kron( S_site[1], np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3),np.eye(3) ) ) ) ) ) ,
np.kron( np.eye(3), np.kron( S_site[1], np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3),np.eye(3) ) ) ) ) ),
np.kron( np.eye(3), np.kron( np.eye(3), np.kron( S_site[1], np.kron( np.eye(3), np.kron( np.eye(3),np.eye(3) ) ) ) ) ),
np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3), np.kron( S_site[1], np.kron( np.eye(3),np.eye(3) ) ) ) ) ),
np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3), np.kron( S_site[1],np.eye(3) ) ) ) ) ),
np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3),S_site[1] ) ) ) ) ) ]

Sz = [ np.kron( S_site[2], np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3),np.eye(3) ) ) ) ) ) ,
np.kron( np.eye(3), np.kron( S_site[2], np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3),np.eye(3) ) ) ) ) ),
np.kron( np.eye(3), np.kron( np.eye(3), np.kron( S_site[2], np.kron( np.eye(3), np.kron( np.eye(3),np.eye(3) ) ) ) ) ),
np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3), np.kron( S_site[2], np.kron( np.eye(3),np.eye(3) ) ) ) ) ),
np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3), np.kron( S_site[2],np.eye(3) ) ) ) ) ),
np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3), np.kron( np.eye(3),S_site[2] ) ) ) ) ) ]


### Commutators check out

### We reorganize the operators 
S = [Sx,Sy,Sz] ### indexed as S[c][r] where c is component and r is site  


### Now we construct the Hamiltonian 
def HAKLT(J0,J1,h):

	out = h[0]*S[2][0] + h[1]*S[2][L-1] ### This couples the end spins to an external field which will be used to interrogate the Loschmidt echos 

	for j in range(L-1):
		
	### Now the AKLT terms 
	### For each neighboring pair of spins this will have -J *(  S[c][j]@S[c][j+1]  )

		pairprod = sum([ S[c][j]@S[c][j+1] for c in range(3) ])

		out +=  J0* ( pairprod + 1./3. * pairprod@pairprod)

		out += J1 * pairprod

	return out


### This computes the Loschmidt echo for the different parameters and final spin projections starting in the ground state 
### tR is the Ramsey time 
def L_echo(J0,J1,h1,h2,tR):
	H1 = HAKLT(J0,J1,h1)
	H2 = HAKLT(J0,J1,h2)

	gs = np.linalg.eigh(HAKLT(J0,J1,[0.,0.]))[1][:,0]

	echo = np.conjugate(gs)@ scp.linalg.expm(1.j*tR*H1) @ scp.linalg.expm(-1.j*tR*H2) @ gs

	return echo



def main():
	J0 = 1.
	J1 = 0.6
	h1 = [0.,0.1]
	h2 = [0.,-0.1]

	nts = 30
	ts = np.linspace(0.,40.,nts)

	echos = np.zeros_like(ts)

	for i in range(nts):
		t = ts[i]
		echos[i] = L_echo(J0,J1,h1,h2,t)

	plt.plot(ts,np.real(echos))
	plt.show()

	J0 = 0.3
	J1 = 1.9
	h1 = [0.,0.1]
	h2 = [0.,-0.1]

	nts = 30
	ts = np.linspace(0.,40.,nts)

	echos = np.zeros_like(ts)

	for i in range(nts):
		t = ts[i]
		echos[i] = L_echo(J0,J1,h1,h2,t)

	plt.plot(ts,np.real(echos))
	plt.show()


if __name__ == "__main__":
	main()










