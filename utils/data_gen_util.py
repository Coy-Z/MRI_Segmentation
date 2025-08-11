import numpy as np
import matplotlib.pyplot as plt

class SDF_MRI():
    '''
    Class to use Signed Distance Function (SDF) to generate MRI data.
    '''
    def __init__(self, V: np.ndarray[float], r: float = 5):
        '''
        Initialize the SDF_MRI class with radius and speed field.
        Args:
            r: radius
            V: speed field
        '''
        self.sdf = np.zeros_like(V)
        self.V = V
        self.a, self.b = V.shape
        self.r = r

        # Set up seed SDF
        # \rho - R for analytical SDF of a circle
        for i in range(self.a):
            for j in range(self.b):
                self.sdf[i, j] = np.sqrt((i - self.a//2)**2 + (j - self.b//2)**2) - r

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
        nabla_neg = np.sqrt(np.maximum(Dn, 0)**2 + np.minimum(Ds, 0)**2 + np.maximum(De, 0)**2 + np.minimum(Dw, 0)**2)

        return nabla_pos, nabla_neg # Order: positive, negative

    def step_sdf(self, iterations: int = 20, dt: float = 1e-2):
        '''
        Perform a time step of the SDF update.
        Args:
            iterations: The number of update iterations.
            dt: The time step size.
        '''
        for _ in range(iterations):
            nablas_pos, nabla_neg = self.get_nablas()
            grad = np.maximum(self.V, 0) * nablas_pos + np.minimum(self.V, 0) * nabla_neg
            self.sdf -= self.V * grad * dt
        return

    def get_sdf(self) -> np.ndarray[float]:
        '''
        Get the current SDF (testing function).
        '''
        return self.sdf

    def activation(self, array: np.ndarray[float], epsilon: float) -> np.ndarray[float]:
        '''
        Apply activation function to the array, to acquire magnitude scan-like behaviour
        '''
        return 0.5 * (1 - np.tanh(3 * array / epsilon))

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

# Testing
V = np.ones((100, 100)) * 5
sdf_mri = SDF_MRI(V, r = 20)
#initial_sdf = sdf_mri.get_sdf().copy()
initial_sdf = np.where(sdf_mri.get_sdf() < 0, 1, 0)
sdf_mri.step_sdf(iterations=100, dt=0.01)
#final_sdf = sdf_mri.get_sdf()
final_sdf = np.where(sdf_mri.get_sdf() < 0, 1, 0)

# Plot Results
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
pcm = []
pcm.append(ax[0].pcolormesh(initial_sdf, cmap='viridis'))
ax[0].set_title('Initial SDF')
ax[0].axis('off')
pcm.append(ax[1].pcolormesh(final_sdf, cmap='viridis'))
ax[1].set_title('Final SDF')
ax[1].axis('off')
fig.colorbar(pcm[0], ax = ax[0], shrink = 0.6)
fig.colorbar(pcm[1], ax = ax[1], shrink = 0.6)

plt.tight_layout()
plt.show()