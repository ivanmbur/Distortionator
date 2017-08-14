Distortionator

README

What is it?

Distortionator is a Python program that aids the analysis of distortions in a two dimensional lattice by calculating the best affine transformations between instances of the lattice and extracting from them useful parameters that characterize the distortion.

How do I install it?

You just need to clone the repository found in https://github.com/ivanmbur/Distortionator !

How do I use it?

To use it you must change the variable data_path to the path your data is in. The data is expected to be a .npy file with the shape (number of points in the lattice, number of instances, 2) such that the number with indices [n, t, i] is the position in the i-th cartesian coordinate in the instance t of spot n. The output of the code is a figure with the plots of the "best" convergences, shears, rotations and translations that take a reference instance (build by averaging all instances) to the instance t vs. t. You must also change the variables title and name to the title you want for the plot and the path (either relative to the folder distortionator.py is in or global) you want to save it to respectively.

Contact

If you have problems or want to make modifications to the source code, you may contact me at ivanmbur@gmail.com

Observations

Although the code is very specific, I must mention that the most powerfull part of it is its method of finding best affine transformations. A detailed explanation of the mathematics behind the method can be downloaded by typing this hrcak.srce.hr/file/1425 into the search bar of an internet browser. In case of need, the reader is invited to take pieces of the code for modification and adding them to their work.
