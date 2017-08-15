import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

data_path = ""
title = ""
plot_name = ""
text_name = ""

#Import chaz cube
chaz_cube = np.load(data_path)

#Build the reference sample
sample_ref = np.average(chaz_cube, axis = 1)

#Build the function which accepts 2 set of N vectors and returns the best affine transformation between them
def affine_fit(reference_raw, target_raw):
	#We first make sure the sets are centered
	center = np.average(reference_raw, axis = 0)
	reference = reference_raw - center
	target = target_raw - center

	#We add a 1 to the end of every vector in reference
	reference_extended = []
	for vector in reference:
		reference_extended.append(np.append(vector, [1]))
	reference_extended = np.array(reference_extended)

	#We calculate the Q matrix
	matrix_list = []
	for vector in reference_extended:
		vector_2D_transposed = vector[np.newaxis]
		matrix_list.append(np.dot(vector_2D_transposed.T, vector_2D_transposed))
	Q = np.sum(np.array(matrix_list), axis = 0)

	#We calculate the c matrix
	C = np.dot(target.T, reference_extended)

	#Solve the linear system
	solution = [] 
	for c in C:
		solution.append(np.linalg.solve(Q, c))
	solution = np.array(solution)

	#return solution
	return solution

#Build a function that returns convergence, shear, rotation and displacement 
def distortion_parameters(A):
    return np.append(np.linalg.solve(np.array([[-1, -1, 0, 0], [0, 0, -1, -1], [0, 0, -1, 1], [-1, 1, 0, 0]]), np.array([A[0,0]-1, A[0,1], A[1,0], A[1,1]-1])), [A[0,2], A[1,2]])

#Create time series of each weak variable
weak_time_series = []
for n in range(0,len(chaz_cube[0])):
	weak_time_series.append(distortion_parameters(affine_fit(sample_ref, chaz_cube[:,n])))
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
fig.savefig(plot_name)
plt.close(fig)

np.savetxt(text_name, weak_time_series)


