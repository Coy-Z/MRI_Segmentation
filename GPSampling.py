# Code written by Matthew Yoko to demonstrate untrained Gaussian Process sampling.

import numpy as np
import matplotlib.pyplot as plt
import time

# To make this fast, we exploit the fact that we are working on a structured grid.
# The RBF kernel becomes seperable on a structured grid, i.e. K = kron(Ky,Kx),
# so we only need to compute the RBF on two 1D grids, then take their Kroneker
# product to form the full 2D kernel, which is much cheaper than computing the 
# 2D RBF kernel directly.
# 
# The second speedup comes from recognising that we really want the Cholesky
# decomposed kernel, L, which we use to sample from the GP. Again, rather 
# than building K and then computing L, we recognize that the problem 
# is separable and so we can Cholesky decompose Kx and Ky, then draw
# our samples using Lx @ z @ Ly^T = L @ z

def rbf_1d(x, lengthscale, variance):
    # Compute euclidian distances in this direction
    dx2 = (x[:, None] - x[None, :])**2
    # Compute the kernel in this dimension
    K1d = variance * np.exp(-0.5 * dx2 / lengthscale**2)
    # Add a small number to the diagonal to make sure K is SPD to numerical precision
    K1d += np.eye(K1d.shape[0])*1e-8*variance
    # Cholesky-decompose
    L1d = np.linalg.cholesky(K1d) 
    return L1d

# Generate a cartesian grid basis
nx, ny = 100, 100
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
# Set the GP hyperparameters
lengthscale = 0.1 # <- sets spatial correlation
variance = 0.2    # <- sets height of ripples
mean = 0.5        # <- constant vertical offset

# Generate random vector
rng = np.random.default_rng()
z = rng.standard_normal((nx, ny))

# start the clock
t0 = time.time()

# Form separated, decomposed kernel and sample
Lx = rbf_1d(x,lengthscale=lengthscale, variance=variance)
Ly = rbf_1d(y,lengthscale=lengthscale, variance=variance)
sample = mean + Lx @ z @ Ly.T          

t1 = time.time()
print(f"Elapsed time: {t1 - t0}... I'm fast as fuck boi")

im = plt.imshow(sample, vmin=0, vmax=1, extent=(0,1,0,1))
plt.colorbar(im)
plt.show()