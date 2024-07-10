import numpy as np
import time 
from matplotlib import pyplot as plt 


def load_csv(fname):

	data = np.genfromtxt(fname+".csv",delimiter=" ")

	nt = data.shape[0]
	L = int(np.sqrt(data.shape[1]))

	data = data.reshape(nt,L,L)

	return data



data_directory = "../data/"
files = ["vorticity_L=10_t-5000_T=0.500000"]

paths = [data_directory + file for file in files]

data = [ load_csv(path) for path in paths]

plt.imshow(data[0,:,:])
plt.colorbar()
plt.show()

plt.imshow(data[1000,:,:])
plt.colorbar()
plt.show()

plt.imshow(data[4000,:,:])
plt.colorbar()
plt.show()