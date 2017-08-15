Distortionator

README

What is it?

Distortionator is a Python program that aids the analysis of distortions in a two dimensional lattice by calculating the best affine transformations between instances of the lattice and extracting from them useful parameters that characterize the distortion.

How do I install it?

You just need to clone the repository found in https://github.com/ivanmbur/Distortionator !

How do I use it?

To use it you must change the variable data_path to the path your data is in. The data is expected to be a .npy file with the shape (number of points in the lattice, number of instances, 2) such that the number with indices [n, t, i] is the position in the i-th cartesian coordinate in the instance t of point n. The output of the code is a text file of the "best" convergences, shears, rotations and translations that take a reference instance (build by averaging all instances) to the instance t vs. t and a plot of it. An example showing the meaning of these parameters is found in example_2.png . It shows what example.png looks after a transformation is applied with all parameters null except for the one in the title of each plot. Moreover, the legend shows the outcome of the code when applied to the reference lattice shown in example.png and the lattice shown in the plot. This serves as a sanity check for the behavior of the code. You must also change the variables title, plot_name and text_name to the title you want for the plot and the path (either relative to the folder distortionator.py is in or global) you want to save the plot and the text to respectively. An example of the full outcome of the code when applied to a lattice of centroids of spots projected onto a sensor during a exposure that was read at various instances is shown in example_3.png and example_4.txt

Methods

affine_fit(reference_raw, target_raw)
	Outputs the best affine transformation x -> Ax + b that takes the reference_raw to the target_raw (after subtracting the mean of the reference_raw) in a numpy array with the augmented matrix form (A|b)

	reference_raw: numpy array
		The reference lattice in an array with shape (number of points, 2) such that the index [n, i] refers to the i-th index of the point n  

	target_raw: numpy array
		The target lattice in an array with shape (number of points, 2) such that the index [n, i] refers to the i-th index of the point n  

distortion_parameters(A)
	Outputs the distortion parameters in a numpy array [convergence, shear_1, shear_2, rotation, translation_1, translation_2] from the affine transformation A

	A: numpy array
		The affine transformation in format [linear transformation | translation]

Contact

If you have problems or want to make modifications to the source code, you may contact me at ivanmbur@gmail.com

Observations

Although the code is very specific, I must mention that the most powerful part of it is its method of finding best affine transformations. A detailed explanation of the mathematics behind the method can be found in H. Späth, Fitting affine and orthogonal transformations between two sets of points, Mathematical Communications 9(2004), 27-34. In case of need, the reader is invited to take pieces of the code for modification and adding them to their work.

I would like to point out some details on the code:

	- To have good sensitivity to the linear parameters (convergence, shear and rotation) the lattices must be reasonably centered. Nevertheless, if all instances of the lattice are perfectly centered, then you get no sensitivity to translations. Therefore the function for calculating affine transformations centers both the reference and the target lattice by using the mean of the reference lattice.
