### Code to compute stochastic dynamics of XY model and tabulate higher order statistics for vortex non-gaussian noise 
### Jonathan Curtis
### 06/26/2024

import numpy as np
from scipy import stats as sts
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

	return out

### This function generates a simulation which is a set of time-traces of theta for a given set of parameters
### The returned array theta will be a large array of shape 
### Nsample x Ntimes x L x L 
### We also return the vorticity profile of the simulations, calculated on the fly to avoid passing arrays 
### These are returned as a tuple thetas, vorticity 
def run_sim(L,T,nburn,nsample,ntimes,J=1.,dt=0.1):

	out = np.zeros((nsample,ntimes,L,L))
	vort_out = np.zeros((nsample,ntimes,L,L))

	for ns in range(nsample):
		### Every sample has a different initial condition and should therefore improve the independence of the distribution 
		out[ns,0,...] = gen_burned_thetas(L,T,nburn,J,dt)
		vort_out[ns,0,...] = calc_vort(out[ns,0,...])

		for nt in range(1,ntimes):
			out[ns,nt,...] = out[ns,nt-1,...] + J*dt*compact_lattice_derivative(out[ns,nt-1,...]) + np.random.normal(0.,2.*T*dt,size=(L,L))
			vort_out[ns,nt,...] = calc_vort(out[ns,nt,...])

	return out, vort_out

#######################################
### THESE METHODS ARE FOR PROCESSING VORTICITY DATA AND ARE SUITABLE FOR BOTH PYTHON OR C++ OUTPUTS 
#######################################

### This method generates the NV filter function for the passed list of distances 

def gen_NV_mask(L,z_list):

	### First we extract how many z points we will be computing for 
	nzs = len(z_list)

	### We also need the corresponding momentum space points to compute the filters 
	### These are not lattice momenta and therefore need to range from -pi/L to pi/L 
	### This method automatically generates the relevant momenta given the size of array, but we need to still weight by pi since the frequencies here are 0, 1/L, 2/L, ... 
	q_list = np.pi*np.fft.fftfreq(L)

	qx = np.outer(q_list,np.ones(L))
	qy = np.outer(np.ones(L),q_list)

	q = np.sqrt(qx*qx + qy*qy)

	### Now we compute the filter functions, which has one for each z point we want 
	NV_masks = np.zeros((nzs,L,L),dtype=complex)
	NV_masks = np.exp(-np.tensordot(z_list, q, axes=0) )/(2.*np.tensordot(np.ones(nzs), q,axes=0))
	NV_masks[:,0,0] = 0.


	return NV_masks

### Given a simulation of the vorticity dynamics this will compute the NV magnetic field for the given distances 
### Object out has shape nsamples x ntimes x nzs 
### Corresponds to a set of nsamples independent samples of trajectories of magnetic field at distance z_list[nzs] for ntimes
def NV_field(vorticity,z_list):
	### First we extract how many z points we will be computing for 
	nzs = len(z_list)

	### Now we extract the relevant array shapes
	nsamples = vorticity.shape[0] ### Number of samples in passed theta array
	ntimes = vorticity.shape[1] ### Number of time points in each array
	L = vorticity.shape[2] ### Size of system

	### We will need FFT only once for all z points but for each sample and time step
	### We compute the fft but insist it have the same output shape as the input array, for convenience
	ffts = np.fft.fft2(vorticity,axes=(-1,-2))

	nv_masks = gen_NV_mask(L,z_list)

	out = np.zeros((nsamples,ntimes,nzs),dtype=complex)

	### Should it be sum or mean?
	#out = np.sum(ffts * filter_funcs,axes=(-1,-2))
	out = np.real( np.tensordot(ffts , nv_masks,axes=([-1,-2],[-1,-2])) )### We return real value, having confirmed it is indeed real

	return out


### Given an ensemble of field measurements for different times and distances this method will compute the relevant moments
### We compute for a number of different evolution times given in t_lists
### These can be up to half the total sample time 
def NV_moments(vorticity,z_list,time_lists):
	### We first process the field to obtain the ensemble of magnetic fields B_z(z,t)
	b_fields = NV_field(vorticity,z_list)

	print(b_fields.shape)

	### We extract the shape parameters 
	nsamples = b_fields.shape[0]
	total_times = b_fields.shape[1]
	nzs = b_fields.shape[2]

	### Now we compute the time integrals 
	### How many time intervals we compute for
	nts = len(time_lists)

	### For the second moment we only need int_0^T B(t)dt for each T in time_lists
	### For the fourth-moment we also need int_T^2T B(t)dt for each T in time_lists 

	### For now we will implement this via a naive loop over z and t points
	### Presumably these arrays will not be quite as large 
	### It will be good in the future to parallelize this 

	### This will store all the possible moments up to 4th order
	### For two variables we should have in principle 2^k moments at order k 
	### We therefore have 2 + 4 + 8 + 16 = 30 entries 
	### In general this would be 2^(n+1) - 2 = 2(2^n - 1) for nth moment
	### Many moments should in principle be equal but may have sampling error we will want to average out 
	moments = np.zeros((30,nzs,nts))

	for i in range(nzs):
		for j in range(nts):
			t_ramsey = time_lists[j] ### This is the time-point in the evolution the pi pulse is applied
			X = np.zeros((2,nsamples)) ### we will have X_ns = [A_ns,B_ns]
			X[0,:] = np.mean(b_fields[:,:t_ramsey,i],axis=1) ### Phase acquired over [0,T_j] for distance z_j
			X[1,:] = np.mean(b_fields[:,t_ramsey:(2*t_ramsey),i],axis=1) ### Phase acquired over [T_j,2T_j] for distance z_i

			### Now we compute the first few moments of the variables A,B
			moment1 = sts.moment(X,order=1,axis=-1,center=0.)
			moment2 = sts.moment(X,order=2,axis=-1,center=0.)
			moment3 = sts.moment(X,order=3,axis=-1,center=0.)
			moment4 = sts.moment(X,order=4,axis=-1,center=0.)
			print(moment1.shape)
			print(moment2.shape)
			print(moment3.shape)
			print(moment4.shape)





	return None



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

	L = 30
	T = 3.7
	nburn = 100
	nsample = 1000
	ntimes = 200 
	
	z_list = np.array([1.,3.,5.,10.])
	t_list = np.array([20,40,60,80])

	t0 = time.time()

	thetas, vorts = run_sim(L,T,nburn,nsample,ntimes)

	second_moment = NV_moments(vorts,z_list,t_list)
		
	t1 = time.time()
	print(t1-t0,"s")
	for j in range(len(z_list)):

		plt.plot(second_moment[:,j])
		plt.xscale('log')
		plt.yscale('log')
	
	plt.show()



if __name__ == "__main__":
	main()












