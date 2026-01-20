This small repository is a numerical example for solving the 2d multi-frequency inverse source problem on a disc with near field measurements [1]. The problem is discretized with a triangular grid. The forward operator can be written as an integral operator which includes the fundamental solution of the Helmholtz equation. A low order quadrature rule for this integral is used in order to setup a forward operator matrix. Reconstruction is done by calculating the SVD of the forward operator matrix. SVD is calculated on GPU using the cupy library. Cupy can be replaced with numpy.
The code in this repository is based on Matlab code originally written by Adrian Kirkeby.


[1] "Stable source reconstruction from a finite number of measurements in the multi-frequency inverse source problem" by Mirza Karamehmedović, Adrian Kirkeby and Kim Knudsen,
https://iopscience.iop.org/article/10.1088/1361-6420/aaba83.
