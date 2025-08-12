import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, solve_triangular
from scipy.spatial.distance import cdist
import time

class Random_Speed_Field():
    def __init__(self, shape: tuple[int, int]):
        '''
        Initialize the speed field.
        Args:
            shape: The shape of the speed field (height, width).
        '''
        self.V = np.zeros(shape)

    def sinusoidal(self, freq_range: tuple[float], amp_range: tuple[float], num_modes: int = 2):
        '''
        Apply a random sinusoidal modulation to the speed field.
        Args:
            freq_range: The frequency range (min, max) for the sinusoidal modulation.
            amp_range: The amplitude range (min, max) for the sinusoidal modulation.
            num_modes: The number of sinusoidal modes to apply.
        '''
        y, x = np.indices(self.V.shape)
        # Center the origin
        y -= self.V.shape[0] // 2
        x -= self.V.shape[1] // 2
        vec = np.stack((x, y), axis=-1)
        for _ in range(num_modes):
            frequency = np.random.uniform(*freq_range)
            amplitude = np.random.uniform(*amp_range)
            phase = np.random.uniform(0, 2 * np.pi)
            angle = np.random.uniform(0, 2 * np.pi)
            self.V += amplitude * np.sin(2 * np.pi * frequency * (vec @ np.array([np.cos(angle), np.sin(angle)]) + phase))
        return

    def affine(self, grad_range: tuple[float], bias_range: tuple[float]):
        '''
        Apply a random affine modulation to the speed field.
        Args:
            grad_range: The gradient range (min, max) for the affine transformation.
            bias_range: The bias range (min, max) for the affine transformation.

        N.B. Adding several affine transformations yields a net affine transformation, so is useless.
        '''
        y, x = np.indices(self.V.shape)
        vec = np.array([x, y])
        gradient = np.random.uniform(*grad_range, 2)
        bias = np.random.uniform(*bias_range)
        self.V += (vec.T @ gradient).T + bias
        return

    def get_speed_field(self) -> np.ndarray[float]:
        '''
        Get the current speed field.
        '''
        return self.V.copy()

class Level_Set_SDF():
    '''
    Class to use Level Set Iterative Methods to warp SDFs.
    '''
    def __init__(self, V: np.ndarray[float], SDF: np.ndarray[float] = None):
        '''
        Initialize the Level_Set_SDF class with a seed SDF and speed field.
        Args:
            V: Speed field
            SDF: Initial signed distance field (must be the same size as V)
        '''
        self.V = V
        self.dt = min(0.05 / self.V.max(), 0.02) # Time step size based on max speed -> enforces CFL conditions
        # Set up seed SDF
        if SDF is not None:
            self.sdf = SDF
        else:
            self.sdf = np.zeros_like(V)

    def get_derivatives(self) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
        '''
        Get the 4 cardinal derivatives of the SDF.
        Returns:
            Dn, Ds, De, Dw: The 4 cardinal derivatives (North, South, East, West)
        '''
        # Pad SDF edges
        padded_sdf = np.pad(self.sdf, pad_width=1, mode='edge')

        # Calculate Cardinal Derivatives
        Dn = np.zeros_like(self.sdf)
        Dn = (padded_sdf[2:, 1:-1] - padded_sdf[1:-1, 1:-1])

        Ds = np.zeros_like(self.sdf)
        Ds = (padded_sdf[1:-1, 1:-1] - padded_sdf[:-2, 1:-1])

        De = np.zeros_like(self.sdf)
        De = (padded_sdf[1:-1, 2:] - padded_sdf[1:-1, 1:-1])

        Dw = np.zeros_like(self.sdf)
        Dw = (padded_sdf[1:-1, 1:-1] - padded_sdf[1:-1, :-2])

        return Dn, Ds, De, Dw # Order: North South East West

    def get_nablas(self) -> tuple[np.ndarray[float], np.ndarray[float]]:
        '''
        Compute quadrature summed directional gradients (smoothing ridges/troughs).
        This helps us enforce upwinding.
        Returns:
            nabla_pos, nabla_neg: The positive and negative nabla fields (for selection dependent on sgn(V))
        '''
        Dn, Ds, De, Dw = self.get_derivatives()

        nabla_pos = np.zeros_like(self.sdf)
        nabla_neg = np.zeros_like(self.sdf)

        nabla_pos = np.sqrt(np.minimum(Dn, 0)**2 + np.maximum(Ds, 0)**2 + np.minimum(De, 0)**2 + np.maximum(Dw, 0)**2)
        nabla_neg = np.sqrt(np.maximum(Dn, 0)**2 + np.minimum(Ds, 0)**2 + np.maximum(De, 0)**2 + np.minimum(Dw, 0)**2) # double check

        return nabla_pos, nabla_neg # Order: positive, negative

    def step_sdf(self, iterations: int = 20, dt: float = None):
        '''
        Perform a time step of the SDF update.
        Args:
            iterations: The number of update iterations.
            dt: The time step size.
        '''
        if dt is None:
            dt = self.dt

        for _ in range(iterations):
            nablas_pos, nabla_neg = self.get_nablas()
            grad = np.maximum(self.V, 0) * nablas_pos - np.minimum(self.V, 0) * nabla_neg # double check
            self.sdf -= self.V * grad * dt
        return

    def get_sdf(self) -> np.ndarray[float]:
        '''
        Get the current SDF (testing function).
        Returns:
            The current SDF.
        '''
        return self.sdf.copy()


class SDF_MRI(Level_Set_SDF):
    '''
    Daughter class of the Level_Set_SDF class.
    This allows us to turn SDFs into MRI-like data and corresponding segmentation masks.
    '''
    def __init__(self, V: np.ndarray[float], SDF: np.ndarray[float] = None):
        super().__init__(V, SDF)
        self.N = V.shape[0] # We will assume we are always generating square data.

    def activation(self, array: np.ndarray[float], epsilon: float) -> np.ndarray[float]:
        '''
        Apply an activation function to the array to acquire MRI magnitude scan-like behaviour.
        Args:
            array: The input array to apply the activation function to.
            epsilon: A value to determine the boundary layer thickness.
        '''
        return 0.5 * (1 - np.tanh(3 * array / epsilon))
    
    def gaussian_process(self, grid_shape: tuple[int, int] = None, length_scale: float = 1., variance: float = 1.):
        '''
        Sample random smooth functions from an untrained Gaussian Process.
        Args:
            grid_shape: The shape of the grid to sample on (height, width). If None, uses self.V.shape
            length_scale: The length scale of the Gaussian Process (larger -> smoother).
            variance: The variance of the Gaussian Process (controls amplitude of function).
        Returns:
            The sampled function.
        '''
        if grid_shape is None:
            grid_shape = self.V.shape

        # Create coordinate grid
        h, w = grid_shape
        y_coords = np.linspace(0, 1, h)[:, None]
        x_coords = np.linspace(0, 1, w)[None, :]

        # Stack coordinates
        coords = np.stack([
            np.repeat(x_coords, h, axis=0).flatten(),
            np.repeat(y_coords, w, axis=1).flatten()
        ], axis=1)
        n_points = coords.shape[0]

        # Compute pairwise distances
        distances = cdist(coords, coords, metric='euclidean')

        # RBF kernel: K(x, x') = variance * exp{-0.5 * ||x - x'||^2 / length_scale^2}
        K = variance * np.exp(-0.5 * distances**2 / length_scale**2)

        # Add small jitter for numerical stability
        K[np.diag_indices_from(K)] += 1e-6

        # Cholesky decomposition for efficient sampling
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # If Cholesky fails, add more jitter
            K[np.diag_indices_from(K)] += 1e-3
            L = np.linalg.cholesky(K)

        # Sample from standard normal and transform
        z = np.random.normal(0, 1, n_points)
        samples = L @ z

        return samples.reshape(grid_shape)
    
    def gaussian_process_sparse(self, grid_shape: tuple[int, int] = None, length_scale: float = 1., variance: float = 1., n_inducing: int = 100):
        '''
        Fast sparse GP sampling using inducing points for large grids.
        Args:
            grid_shape: The shape of the grid to sample on (height, width). If None, uses self.V.shape
            length_scale: The length scale of the Gaussian Process (larger -> smoother).
            variance: The variance of the Gaussian Process (controls amplitude of function).
            n_inducing: The number of inducing points to use.
        Returns:
            The sampled function.
        '''
        if grid_shape is None:
            grid_shape = self.V.shape

        h, w = grid_shape

        # Create inducing points grid (much smaller)
        n_ind_h = min(int(np.sqrt(n_inducing * h / w)), h)
        n_ind_w = min(int(np.sqrt(n_inducing * w / h)), w)
            
        # Inducing points coordinates
        y_ind = np.linspace(0, 1, n_ind_h)
        x_ind = np.linspace(0, 1, n_ind_w)
        y_ind_grid, x_ind_grid = np.meshgrid(y_ind, x_ind, indexing='ij')
        inducing_coords = np.column_stack([x_ind_grid.ravel(), y_ind_grid.ravel()])
            
        # Full grid coordinates
        y_coords = np.linspace(0, 1, h)
        x_coords = np.linspace(0, 1, w)
        y_full, x_full = np.meshgrid(y_coords, x_coords, indexing='ij')
        full_coords = np.column_stack([x_full.ravel(), y_full.ravel()])
            
        # Compute kernels
        K_uu = variance * np.exp(-0.5 * cdist(inducing_coords, inducing_coords)**2 / length_scale**2)
        K_uu[np.diag_indices_from(K_uu)] += 1e-6
            
        K_uf = variance * np.exp(-0.5 * cdist(inducing_coords, full_coords)**2 / length_scale**2)
            
        # Cholesky of K_uu
        L_uu = np.linalg.cholesky(K_uu)
            
        # Sample inducing values
        z_u = np.random.normal(0, 1, len(inducing_coords))
        u_samples = L_uu @ z_u
        
        # Project to full grid
        A = np.linalg.solve(L_uu, K_uf)
        samples = A.T @ np.linalg.solve(L_uu, u_samples)

        return samples.reshape(grid_shape)

    def add_noise(self, arr: np.ndarray[float], noise_level: float = 1.) -> np.ndarray[float]: # Incomplete
        '''
        Add fine Gaussian noise and smooth noise (sampled Gaussian Process) to the SDF.
        Args:
            arr: The input array to add noise to.
            noise_level: The level of noise to add (order-of-magnitude).
        '''
        white_noise = np.random.normal(0, noise_level/5, arr.shape)
        if arr.shape[0] * arr.shape[1] > 1600:
            smooth_noise = self.gaussian_process_sparse(grid_shape = arr.shape, length_scale = self.N / 1000, variance = noise_level)
        else:
            smooth_noise = self.gaussian_process(grid_shape = arr.shape, length_scale = self.N / 1000, variance = noise_level)
        arr += white_noise + smooth_noise
        return

    def return_mask_magn_pair(self):
        '''
        Get the mask and magnitude pair from the SDF.
        Returns:
            A tuple containing the mask and magnitude arrays.
        '''
        mask = np.where(self.sdf.copy() < 0, 1, 0)
        magn = self.activation(self.sdf.copy(), epsilon = 15) * 10 # Magnitude ~ 10
        self.add_noise(magn, noise_level = 1)  # Add noise to the magnitude ~ 1
        return mask, magn

class SDF_MRI_Circle(SDF_MRI):
    '''
    Daughter class of the SDF_MRI class.
    Initialises seed SDF for a circle.
    '''
    def __init__(self, V: np.ndarray[float], r: float = 5, center_var: float = 0.1):
        '''
        Initialize the SDF_MRI_Circle class with speed field, radius and center position variance.
        There are several caveats with the generation of circle center position:
        1. The covariance matrix is a multiple of the identity since we wish for rotational invariance.
        2. The Gaussian distribution is used for center position sampling, so we get more samples near the center.
           Consider that the majority of MRI scans will be centered on the point of interest, edge cases are rarer.
        3. We do however need some edge cases, and even some cases where the center does not lie in the FOV.
           This is achieved via the tails of the Gaussian.
        4. Empty images should also exist in the training set.
        Args:
            r: The circle radius.
            V: The speed field.
            center_var: The variance for center position sampling.
        '''
        super().__init__(V, SDF = None)
        # Set up seed SDF
        # \rho - R for analytical SDF of a circle
        # Randomise center location for location invariance
        center = np.random.multivariate_normal(mean=np.zeros(2), cov=np.eye(2) * center_var) * V.shape // 2 + np.array(V.shape) // 2
        vec = np.stack(np.indices(V.shape), axis = -1)
        self.sdf = np.linalg.norm(vec - center, axis=-1) - r  # Ensure SDF is non-negative outside the circle

class SDF_MRI_Tube(SDF_MRI):
    '''
    Daughter class of the SDF_MRI class.
    Initialises seed SDF for a tube.
    '''
    def __init__(self, V: np.ndarray[float], r: float = 10., dir: tuple[float, float] = None, point_var: float = 0.1):
        '''
        Initialize the SDF_MRI_Tube class with speed field, direction, point position variance, and radius.
        There are several caveats with the generation of tube point position:
        1. The covariance matrix is a multiple of the identity since we wish for rotational invariance.
        2. The Gaussian distribution is used for point position sampling, so we get samples with significant tube area.
           Consider that the majority of MRI scans will be centered on the point of interest, edge cases are rarer.
        3. We do however need some edge cases, and even some cases where the center does not lie in the FOV.
           This is achieved via the tails of the Gaussian.
        4. Empty images should also exist in the training set.
        Args:
            dir: The direction vector of the tube. If None, a random direction will be used.
            point_var: The variance for point position sampling (to pin the tube).
            r: The tube radius.
        '''
        super().__init__(V, SDF = None)
        # Set up seed SDF
        # (r - p).n - t for analytical SDF of a tube
        # Find normal to dir
        if dir is not None:
            n = np.array([-dir[1], dir[0]])
            n /= np.linalg.norm(n)  # Normalize the normal vector
        else:
            angle = np.random.uniform(0, 2 * np.pi)
            n = np.array([np.cos(angle), np.sin(angle)])
        point = np.random.multivariate_normal(mean=np.zeros(2), cov=np.eye(2) * point_var) * V.shape // 2 + np.array(V.shape) // 2
        vec = np.stack(np.indices(V.shape), axis=-1)
        # Compute the signed distance function
        self.sdf = np.abs(np.dot(vec - point, n)) - r  # SDF is the distance to the tube

# ----- Testing -----

since = time.time()
V = Random_Speed_Field(shape=(100, 100))
V.sinusoidal(freq_range=(0.001, 0.05), amp_range=(0, 1), num_modes=2)
V.affine(grad_range=(-0.05, 0.05), bias_range=(-1, 2))
sdf_mri_circ = SDF_MRI_Circle(V.get_speed_field(), r = 22)
sdf_mri_tube = SDF_MRI_Tube(V.get_speed_field(), r=10.)

#smooth_noise = sdf_mri_circ.gaussian_process(grid_shape=(100, 100), length_scale=0.1, variance=10.)
#initial_sdf = sdf_mri_circ.get_sdf()
#initial_sdf = np.where(sdf_mri_circ.get_sdf() < 0, 1, 0)

sdf_mri_circ.step_sdf(iterations=100)
sdf_mri_tube.step_sdf(iterations=100)
maskc, magnc = sdf_mri_circ.return_mask_magn_pair()
maskt, magnt = sdf_mri_tube.return_mask_magn_pair()

#final_sdf = sdf_mri_circ.get_sdf()

print("Time taken:", time.time() - since)

# Plot Results
fig, ax = plt.subplots(2, 2, figsize=(8, 6))
pcm = []
pcm.append(ax[0, 0].pcolormesh(maskc, cmap='viridis'))
ax[0, 0].set_title('Mask')
ax[0, 0].axis('off')
pcm.append(ax[0, 1].pcolormesh(magnc, cmap='viridis'))
ax[0, 1].set_title('Magnitude')
ax[0, 1].axis('off')
pcm.append(ax[1, 0].pcolormesh(maskt, cmap='viridis'))
ax[1, 0].set_title('Mask')
ax[1, 0].axis('off')
pcm.append(ax[1, 1].pcolormesh(magnt, cmap='viridis'))
ax[1, 1].set_title('Magnitude')
ax[1, 1].axis('off')
fig.colorbar(pcm[0], ax = ax[0, 0], shrink = 0.6)
fig.colorbar(pcm[1], ax = ax[0, 1], shrink = 0.6)
fig.colorbar(pcm[2], ax = ax[1, 0], shrink = 0.6)
fig.colorbar(pcm[3], ax = ax[1, 1], shrink = 0.6)

plt.tight_layout()
plt.show()