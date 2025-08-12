import numpy as np
import matplotlib.pyplot as plt
import time

class SDF_MRI():
    '''
    Class to use Signed Distance Function (SDF) to generate MRI data.
    '''
    def __init__(self, V: np.ndarray[float], r: float = 5, centre_var: float = 0.5):
        '''
        Initialize the SDF_MRI class with radius and speed field.
        Args:
            r: radius
            V: speed field
        '''
        self.sdf = np.zeros_like(V)
        self.V = V
        self.r = r
        self.dt = min(0.05 / self.V.max(), 0.02) # Time step size based on max speed -> enforces CFL conditions
        
        # Set up seed SDF
        # \rho - R for analytical SDF of a circle
        # Randomise centre location for location invariance
        centre = np.random.multivariate_normal(mean=np.zeros(2), cov=np.eye(2) * centre_var) * V.shape // 2 + np.array(V.shape) // 2
        vec = np.stack(np.indices(V.shape), axis = -1)
        self.sdf = np.linalg.norm(vec - centre, axis=-1) - r  # Ensure SDF is non-negative outside the circle

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
        Compute quadrature summed directional gradients, ensuring upwinding.
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
        '''
        return self.sdf.copy()

    def activation(self, array: np.ndarray[float], epsilon: float) -> np.ndarray[float]:
        '''
        Apply activation function to the array, to acquire magnitude scan-like behaviour
        '''
        return 0.5 * (1 - np.tanh(3 * array / epsilon))
    
    def add_smoothing(self, sigma: float) -> np.ndarray[float]: # Incomplete
        return

    def add_noise(self, noise_level: float) -> np.ndarray[float]: # Incomplete
        '''
        Add Gaussian noise to the SDF.
        '''
        noise = np.random.normal(0, noise_level, self.sdf.shape)
        self.sdf += noise
        return self.sdf

    def return_mask_magn_pair(self):
        '''
        Get the mask and magnitude pair from the SDF.
        '''
        mask = np.where(self.sdf < 0, 1, 0)
        magn = self.activation(self.sdf, epsilon = 2/self.a)
        return mask, magn
    
class Speed_Field():
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

# ----- Testing -----

V = Speed_Field(shape=(100, 120))
V.sinusoidal(freq_range=(0.001, 0.05), amp_range=(0.5, 1.5), num_modes=2)
V.affine(grad_range=(-0.05, 0.05), bias_range=(-1, 1))
sdf_mri = SDF_MRI(V.get_speed_field(), r = 22)
#initial_sdf = sdf_mri.get_sdf()
initial_sdf = np.where(sdf_mri.get_sdf() < 0, 1, 0)
sdf_mri.step_sdf(iterations=100)
#final_sdf = sdf_mri.get_sdf()
final_sdf = np.where(sdf_mri.get_sdf() < 0, 1, 0)

# Plot Results
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
pcm = []
pcm.append(ax[0].pcolormesh(initial_sdf, cmap='viridis'))
ax[0].set_title('Initial SDF')
ax[0].axis('off')
pcm.append(ax[1].pcolormesh(final_sdf, cmap='viridis'))
ax[1].set_title('Final SDF')
ax[1].axis('off')
pcm.append(ax[2].pcolormesh(V.get_speed_field(), cmap='viridis'))
ax[2].set_title('Speed Field')
ax[2].axis('off')
fig.colorbar(pcm[0], ax = ax[0], shrink = 0.6)
fig.colorbar(pcm[1], ax = ax[1], shrink = 0.6)
fig.colorbar(pcm[2], ax = ax[2], shrink = 0.6)

plt.tight_layout()
plt.show()