### Code to compute stochastic dynamics of XY model and tabulate higher order statistics for vortex non-gaussian noise 
### Jonathan Curtis
### 06/26/2024

import numpy as np
from scipy import stats as sts
from matplotlib import pyplot as plt 
import time 
import datetime

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

### This computes up to fourth order multivariate moments of variable X[a,nsamples] = [X[0,nsamples].X[1,nsamples]] 
def calc_moments(X):
	out = [np.zeros(2),np.zeros((2,2)),np.zeros((2,2,2)),np.zeros((2,2,2,2))]

	for j in range(2):
		out[0][j] = np.mean(X[j,:])

		for k in range(2):
			out[1][j,k] = np.mean(X[j,:]*X[k,:])

			for l in range(2):
				out[2][j,k,l] = np.mean(X[j,:]*X[k,:]*X[l,:])

				for m in range(2):
					out[3][j,k,l,m] = np.mean(X[j,:]*X[k,:]*X[l,:]*X[m,:])

	return out

### This computes the cumulants given the raw moments 
### Assumes of the form X = [ moment1[2], moment2[2,2], ...]
def calc_cumulants(X):
	out = [ np.zeros(2),np.zeros((2,2)),np.zeros((2,2,2)),np.zeros((2,2,2,2))]

	out[0][:] = X[0][:] ### Mean
	out[1][:,:] = X[1][:,:] - np.tensordot(X[0][:],X[0][:],axes=0) ### Second cumulant
	out[2][:,:,:] = X[2][:,:,:] - 6.*np.tensordot(X[0][:],X[1][:,:],axes=0) + np.tensordot(X[0][:],np.tensordot( X[0][:],X[0][:],axes=0 ),axes=0 )  ### Third cumulant

	out[3][:,:,:,:] = X[3][:,:,:,:] - 4.* np.tensordot(X[0][:],X[2][:,:,:],axes=0) + 6.* np.tensordot( np.tensordot(X[0][:],X[0][:],axes=0), X[1][:,:] ,axes=0) - 3.*np.tensordot(X[0][:],np.tensordot(X[0][:],X[0][:],axes=0),axes=0)
	out[3][:,:,:,:] += -3.*np.tensordot(out[1],out[1],axes=0) ### The fourth cumulant is not the same as the fourth centered moment and has an additional contribution from covariance squared

	return out

### Given an ensemble of field measurements for different times and distances this method will compute the relevant moments
### We compute for a number of different evolution times given in t_lists
### These can be up to half the total sample time 
def NV_moments(vorticity,z_list,time_lists):
	### We first process the field to obtain the ensemble of magnetic fields B_z(z,t)
	b_fields = NV_field(vorticity,z_list)

	### We extract the shape parameters 
	nsamples = b_fields.shape[0]
	total_times = b_fields.shape[1]
	nzs = b_fields.shape[2]

	### Now we compute the time integrals 
	### How many time intervals we compute for
	nts = len(time_lists)

	### For now we will implement this via a naive loop over z and t points
	### Presumably these arrays will not be quite as large 
	### It will be good in the future to parallelize this 

	### We compute the moments up to fourth order
	### We assume odd moments vanish ---- this should be relaxed and controlled for but at the moment it is hard to extract all moments

	### For the time being we extract only the average, full covariance matrix, and the covariance matrix of X^2 , Y^2 at fourth order
	moments = [np.zeros((2,nts,nzs)),np.zeros((2,2,nts,nzs)),np.zeros((2,2,2,nts,nzs)),np.zeros((2,2,2,2,nts,nzs))]

	for i in range(nzs):
		for j in range(nts):
			t_ramsey = time_lists[j] ### This is the time-point in the evolution the pi pulse is applied
			X = np.zeros((2,nsamples)) ### we will have X_ns = [A_ns,B_ns]
			X[0,:] = np.mean(b_fields[:,:t_ramsey,i],axis=1) ### Phase acquired over [0,T_j] for distance z_j
			X[1,:] = np.mean(b_fields[:,t_ramsey:(2*t_ramsey),i],axis=1) ### Phase acquired over [T_j,2T_j] for distance z_i

			M = calc_moments(X)
			#for a in range(2):
			#	for b in range(2):
			#		moments[1][a,b,j,i] = np.mean(X[a,:]*X[b,:])
			#		moments[2][a,b,j,i] = np.mean(X[a,:]*X[a,:]*X[b,:]*X[b,:])

			for a in range(4):
					moments[a][...,j,i] = M[a][...]

	return moments

### Same as NV_moments but returns the cumulants instead
def NV_cumulants(vorticity,z_list,time_lists):
	### We first process the field to obtain the ensemble of magnetic fields B_z(z,t)
	b_fields = NV_field(vorticity,z_list)

	### We extract the shape parameters 
	nsamples = b_fields.shape[0]
	total_times = b_fields.shape[1]
	nzs = b_fields.shape[2]

	### Now we compute the time integrals 
	### How many time intervals we compute for
	nts = len(time_lists)

	### For now we will implement this via a naive loop over z and t points
	### Presumably these arrays will not be quite as large 
	### It will be good in the future to parallelize this 

	### We compute the moments up to fourth order
	### We assume odd moments vanish ---- this should be relaxed and controlled for but at the moment it is hard to extract all moments

	### For the time being we extract only the average, full covariance matrix, and the covariance matrix of X^2 , Y^2 at fourth order
	cumulants = [np.zeros((2,nts,nzs)),np.zeros((2,2,nts,nzs)),np.zeros((2,2,2,nts,nzs)),np.zeros((2,2,2,2,nts,nzs))]

	for i in range(nzs):
		for j in range(nts):
			t_ramsey = time_lists[j] ### This is the time-point in the evolution the pi pulse is applied
			X = np.zeros((2,nsamples)) ### we will have X_ns = [A_ns,B_ns]
			X[0,:] = np.mean(b_fields[:,:t_ramsey,i],axis=1) ### Phase acquired over [0,T_j] for distance z_j
			X[1,:] = np.mean(b_fields[:,t_ramsey:(2*t_ramsey),i],axis=1) ### Phase acquired over [T_j,2T_j] for distance z_i

			M = calc_moments(X)
			C = calc_cumulants(M)

			for a in range(4):
					cumulants[a][...,j,i] = C[a][...]

	return cumulants

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


#######################################
### THIS CLASS WILL WRAP THE SIMULATION WITH I/O TO FACILITATE SYSTEMATIC STUDY
#######################################

### We will assume the run file is formatted as the following:
"""
data_directory = <RELATIVE PATH TO DIRECTORY TO STORE DATA>
run_name = <STRING IDENTIFIER FOR SUB RUN>
save_thetas = <bool>
save_vorticty = <bool>
save_cumulants = <bool>
save_meta = <bool> 
L = <int>
T = <float>
nburn = <int>
nsample = <int>
ntimes = <int>
z_list = <float list, comma separated>
t_list = <int list, comma separated> 
""" 

class run_from_file:
	### This method will read in all the appropriate inputs and organize them into attributes
	def __init__(self,run_file_name):
		self.run_file_name = run_file_name

		with open(self.run_file_name,"r") as f:
			lines = f.readlines()
			### DO THIS WITH A DICTIONARY IN FUTURE
			### Now we parse each line
			self.data_directory = ( (lines[0]).split(' ')[-1]).strip('\n')
			self.run_name = ( (lines[1]).split(' ')[-1]).strip('\n')

			self.save_thetas = bool( (lines[2]).split(' ')[-1] )
			self.save_vorticity = bool( (lines[3]).split(' ')[-1] )
			self.save_cumulants = bool( (lines[4]).split(' ')[-1] )
			self.save_meta = bool( (lines[5]).split(' ')[-1] )

			self.L = int( (lines[6]).split(' ')[-1] )
			self.T = float( (lines[7]).split(' ')[-1] )
			self.nburn = int( (lines[8]).split(' ')[-1] )
			self.nsample = int( (lines[9]).split(' ')[-1] )
			self.ntimes = int( (lines[10]).split(' ')[-1] )

			z_list = ((lines[11]).split(' ')[-1]).strip('\n')
			self.z_list = np.array([ int(z) for z in z_list.split(',') ])

			t_list = ((lines[12]).split(' ')[-1]).strip('\n')
			self.t_list = np.array([ int(t) for t in t_list.split(',') ])

		### This is where we will put the elapsed times taken to run the simulation and compute the cumulants 
		self.run_time = 0.
		self.cumulant_time = 0.

		### It may be useful to timestamp so we also create an attribute to store the time stamp of when the run began
		self.time_stamp = None

		### How many z and t_ramsey points we compute for 
		self.nzs = len(self.z_list)
		self.nts = len(self.t_list)

		### This is where the data will actually be stored 
		self.thetas = np.zeros((self.nsample,self.ntimes,self.L,self.L))
		self.vorticity = np.zeros((self.nsample,self.ntimes,self.L,self.L))

		### We store the moments up to fourth order and keep all cross moments which gives 2 + 4 + 8 + 16 = 30 moments 
		self.cumulants = [np.zeros((2,self.nts,self.nzs)),np.zeros((2,2,self.nts,self.nzs)),np.zeros((2,2,2,self.nts,self.nzs)),np.zeros((2,2,2,2,self.nts,self.nzs))]

	### Calling this method will run the simulation 
	def run(self):
		t0 = time.time()

		self.time_stamp = datetime.datetime.now()

		self.thetas, self.vorticity = run_sim(self.L, self.T, self.nburn, self.nsample, self.ntimes)

		t1 = time.time()

		self.run_time = t1 - t0 ### Elapsed time to run simulation

	### Calling this method will compute the cumulants after running the simulation
	def cumulant(self):
		t0 = time.time()

		C = NV_cumulants(self.vorticity,self.z_list,self.t_list) ### This computes the cumulants and stores them
		for i in range(4):
			self.cumulants[i] = C[i][...]

		t1 = time.time()

		self.cumulant_time = t1 - t0 ### Elapsed time for cumulants 

	### Calling this method will save the data (if indicated)
	def save_data(self):

		if self.save_thetas:
			path = self.data_directory + self.run_name+"_thet.npy"
			np.save(path,self.thetas)

		if self.save_vorticity:
			path = self.data_directory+ self.run_name + "_vort.npy"
			np.save(path,self.vorticity)

		if self.save_cumulants:
			for i in range(len(self.cumulants)):
				### We save each cumulant order to a separate file which is labeled by the cumulant order
				path = self.data_directory + self.run_name+"_cuml_"+str(i)+".npy"
				np.save(path,self.cumulants[i])

		if self.save_meta:
			path = self.data_directory+self.run_name+"_meta.txt"
			with open(path,"w") as f:
				f.write("data_directory = "+self.data_directory+'\n')
				f.write("run_name = "+self.run_name+'\n')
				f.write("save_thetas = "+str(self.save_thetas)+'\n')
				f.write("save_vorticity = "+str(self.save_vorticity)+'\n')
				f.write("save_cumulants = "+str(self.save_vorticity)+'\n')
				f.write("save_meta = "+str(self.save_meta)+'\n')
				f.write("L = "+str(self.L)+'\n')
				f.write("T = "+str(self.T)+'\n')
				f.write("nburn = "+str(self.nburn)+'\n')
				f.write("nsample = "+str(self.nsample)+'\n')
				f.write("ntimes = "+str(self.ntimes)+'\n')
				f.write("z_list = "+(str(self.z_list)[1:-1]).replace(" ",",")+'\n')
				f.write("t_list = "+(str(self.t_list)[1:-1]).replace(" ",",")+'\n')
				f.write('\n')
				f.write("time_stamp = "+str(self.time_stamp)+'\n')
				f.write("run_time = "+str(self.run_time)+'\n')
				f.write("cumulant_time = "+str(self.cumulant_time)+'\n')
				f.close()


def main():

	file_path = "../data/08052024/01.txt"
	run = run_from_file(file_path)
	run.run()
	run.cumulant()
	run.save_data()


if __name__ == "__main__":
	main()












