import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

data_path = "/projector/cshapiro/analysis/2017-07-21/cleanpos/positions_clean_cds-f44spots_pupilbaf-_10_39_mega.npy"
title = "Distortion parameters positions_clean_cds-f44spots_pupilbaf-_10_39_mega"
name = "positions_clean_cds-f44spots_pupilbaf-_10_39_mega_parameters.png"

#Import chaz cube
chaz_cube = np.load(data_path)

#Build the reference sample
sample_ref = np.average(chaz_cube, axis = 1)

#Build the function which accepts 2 set of N vectors and returns the best affine transformation between them
def affine_fit(set_1_raw, set_2_raw):
	#We first make sure the sets are centered
	center = np.average(set_1_raw, axis = 0)
	set_1 = set_1_raw - center
	set_2 = set_2_raw - center

	#We add a 1 to the end of every vector in set_1
	set_1_extended = []
	for vector in set_1:
		set_1_extended.append(np.append(vector, [1]))
	set_1_extended = np.array(set_1_extended)

	#We calculate the Q matrix
	matrix_list = []
	for vector in set_1_extended:
		vector_2D_transposed = vector[np.newaxis]
		matrix_list.append(np.dot(vector_2D_transposed.T, vector_2D_transposed))
	Q = np.sum(np.array(matrix_list), axis = 0)

	#We calculate the c matrix
	C = np.dot(set_2.T, set_1_extended)

	#Solve the linear system
	solution = [] 
	for c in C:
		solution.append(np.linalg.solve(Q, c))
	solution = np.array(solution)

	#return solution
	return solution

#Build a function that returns convergence, shear, rotation and displacement 
def weak_variables(A):
    return np.append(np.linalg.solve(np.array([[-1, -1, 0, 0], [0, 0, -1, -1], [0, 0, -1, 1], [-1, 1, 0, 0]]), np.array([A[0,0]-1, A[0,1], A[1,0], A[1,1]-1])), [A[0,2], A[1,2]])

#Create time series of each weak variable
weak_time_series = []
for n in range(0,len(chaz_cube[0])):
	weak_time_series.append(weak_variables(affine_fit(sample_ref, chaz_cube[:,n])))
weak_time_series = np.array(weak_time_series)

fig, ax = plt.subplots(len(weak_time_series[0]), figsize = (10, 15), sharex = True)
for i in range(0, len(ax)):
	ax[i].plot(range(1, len(weak_time_series) + 1), weak_time_series[:,i], label = "variance %E" % Decimal(np.var(weak_time_series[:,i])))
	ax[i].legend()
ax[0].set_title(title)
ax[-1].set_xlabel("sample number")
ax[0].set_ylabel("convergence ($\kappa$)")
ax[1].set_ylabel("shear ($\gamma_1$)")
ax[2].set_ylabel("shear ($\gamma_2$)")
ax[3].set_ylabel("rotation ($\omega$)")
ax[4].set_ylabel("translation ($b_1$)")
ax[5].set_ylabel("translation ($b_2$)")
plt.tight_layout()
fig.savefig(name)
plt.close(fig)


