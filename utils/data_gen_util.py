import numpy as np

class Random_Speed_Field():
    '''
    Class for initializing a speed field and applying random perturbations.
    
    Improvements to be made:
        - Add warping
        - Add dunder method implementation
    '''
    def __init__(self, shape : tuple[int, int]):
        '''
        Initialize the speed field.
        Args:
            shape (tuple): The shape of the speed field (height, width).
        '''
        self.field = np.zeros(shape)
        self.shape = shape

    def sinusoidal(self, freq_range : tuple[float, float], amp_range : tuple[float, float], num_modes : int = 2):
        '''
        Apply a random sinusoidal modulation to the speed field.
        Args:
            freq_range (tuple): The frequency range (min, max) for the sinusoidal modulation. A log-uniform distribution is then applied.
            amp_range (tuple): The amplitude range (min, max) for the sinusoidal modulation. A uniform distribution is then applied.
            num_modes (int): The number of sinusoidal modes to apply.
        '''
        y, x = np.indices(self.field.shape)
        # Center the origin
        y -= self.field.shape[0] // 2
        x -= self.field.shape[1] // 2
        vec = np.stack((x, y), axis=-1)
        for _ in range(num_modes):
            logfrequency = np.random.uniform(*np.log(np.array(freq_range)))
            frequency = np.exp(logfrequency)
            amplitude = np.random.uniform(*amp_range)
            phase = np.random.uniform(0, 2 * np.pi)
            angle = np.random.uniform(0, 2 * np.pi)
            self.field += amplitude * np.sin(2 * np.pi * frequency * (vec @ np.array([np.cos(angle), np.sin(angle)]) + phase))
        return

    def affine(self, grad_range : tuple[float, float], bias_range : tuple[float, float]):
        '''
        Apply a random affine modulation to the speed field.
        Args:
            grad_range (tuple): The gradient range (min, max) for the affine transformation. A uniform distribution is then applied.
            bias_range (tuple): The bias range (min, max) for the affine transformation. A uniform distribution is then applied.

        N.B. Adding several affine transformations yields a net affine transformation, so is useless.
        '''
        y, x = np.indices(self.field.shape)
        vec = np.array([x, y])
        gradient = np.random.uniform(*grad_range, 2)
        bias = np.random.uniform(*bias_range)
        self.field += (vec.T @ gradient).T + bias
        return

    def cholesky_rbf_1d(self, x : np.ndarray[float], length_scale : float, variance : float) -> np.ndarray[float]:
        '''
        Calculate the Cholesky decomposition of the 1D Radial Basis Function (RBF) kernel.
        Args:
            x (np.ndarray): The input array.
            length_scale (float): The length scale of the RBF.
            variance (float): The variance of the RBF.
        Returns:
            The Cholesky decomposition of the 1D RBF kernel (np.ndarray).
        '''
        # Compute euclidian distances in this direction
        # Note: x[:, None] turns x into a column vector, and x[None, :] turns x into a row vector.
        #       Subtracting one from the other gives the pairwise differences (via broadcasting).
        distances2 = (x[:, None] - x[None, :])**2
        # Compute the kernel in this dimension
        K = variance * np.exp(-0.5 * distances2 / length_scale**2)
        # Add small jitter to the diagonal to make sure K is SPD to numerical precision
        K += np.eye(K.shape[0])*1e-8*variance
        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # If Cholesky fails, add more jitter
            K += np.eye(K.shape[0])*1e-5*variance
            L = np.linalg.cholesky(K)
        return L

    def gaussian_process(self, grid_shape : tuple[int, int] = None, length_scale : float = 1.,
                         variance : float = 1.) -> np.ndarray[float]:
        '''
        Sample random smooth functions from an untrained Gaussian Process. We exploit the separable nature
        of the RBF kernel on a structured grid, i.e. K = Kronecker(K_x, K_y), hence only requiring compute in 1D.
        Args:
            grid_shape (tuple): The shape of the grid to sample on (height, width). If None, uses self.V.shape
            length_scale (float): The length scale of the Gaussian Process (larger -> smoother).
            variance (float): The variance of the Gaussian Process (controls amplitude of function).
        Returns:
            The sampled function (np.ndarray).
        '''
        if grid_shape is None:
            grid_shape = self.shape

        # Create coordinate grid
        ny, nx = grid_shape
        y = np.linspace(0, 1, ny)
        x = np.linspace(0, 1, nx)

        Ly = self.cholesky_rbf_1d(y, length_scale=length_scale, variance=variance)
        Lx = self.cholesky_rbf_1d(x, length_scale=length_scale, variance=variance)

        # Sample from standard normal and transform
        # Generate a random latent vector
        rand_gen = np.random.default_rng()
        z = rand_gen.standard_normal((nx, ny))
        samples = Lx @ z @ Ly.T
        return samples

    def random_coherent(self, log_length_scale_mean : float = 1., log_length_scale_variance : float = 1., amplitude_variance : float = 1.):
        '''
        Add a random coherent photo (sampled from an untrained Gaussian Process).
        Args:
            length_scale_mean (float): The mean length scale for the Gaussian Process (larger -> smoother).
            length_scale_variance (float): The variance of the length scale for the Gaussian Process.
            variance (float): The variance for the Gaussian Process (controls amplitude of function).

        N.B. We use a log-normal distribution for the length scale to ensure positivity,
            and also since it makes more physical sense
        '''
        length_scale = np.exp(np.random.normal(log_length_scale_mean, log_length_scale_variance))
        self.field += self.gaussian_process(grid_shape=self.shape, length_scale=length_scale, variance=amplitude_variance)
        return
    
    def reset(self):
        '''
        Reset the speed field to zero.
        '''
        self.field = np.zeros_like(self.field)

class Level_Set_SDF():
    '''
    Class to use Level Set Iterative Methods to warp SDFs.
    '''
    def __init__(self, V : Random_Speed_Field, SDF : np.ndarray[float] = None):
        '''
        Initialize the Level_Set_SDF class with a seed SDF and speed field.
        Args:
            V (Random_Speed_Field): Speed field
            SDF (np.ndarray or None): Initial signed distance field (must be the same size as V)
        '''
        self.V = V
        self.dt = min(0.05 / self.V.field.max(), 0.02) # Time step size based on max speed -> enforces CFL conditions. Courant Number <= 0.05
        # Set up seed SDF
        if SDF is not None:
            self.sdf = SDF
        else:
            self.sdf = np.zeros_like(V)

    def copy(self) -> 'Level_Set_SDF':
        '''
        Deep copy method for Level_Set_SDF.
        Returns:
            A deep copy of the Level_Set_SDF instance (Level_Set_SDF).
        '''
        level_set_copy = Level_Set_SDF(self.V, self.sdf)
        return level_set_copy

    def update_speed_field(self, V : Random_Speed_Field):
        '''
        Update the speed field associated with the Level_Set_SDF instance.
        Args:
            V (Random_Speed_Field): New speed field
        '''
        assert V.field.shape == self.V.field.shape, "New speed field must be the same shape as current speed field."
        self.V = V
        self.dt = min(0.05 / self.V.field.max(), 0.02) # Update time step size

    def get_derivatives(self) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
        '''
        Get the 4 cardinal derivatives of the SDF.
        Returns:
            Dn, Ds, De, Dw (tuple): The 4 cardinal derivatives (North, South, East, West)
        '''
        # Pad SDF edges
        padded_sdf = np.pad(self.sdf, pad_width=1, mode='edge')

        # Calculate Cardinal Derivatives
        Dn = np.zeros_like(self.sdf)
        Dn = (padded_sdf[2:, 1:-1] - padded_sdf[1:-1, 1:-1])
        Dn[-1] = Dn[-2]

        Ds = np.zeros_like(self.sdf)
        Ds = (padded_sdf[1:-1, 1:-1] - padded_sdf[:-2, 1:-1])
        Ds[0] = Ds[1]

        De = np.zeros_like(self.sdf)
        De = (padded_sdf[1:-1, 2:] - padded_sdf[1:-1, 1:-1])
        De[:, -1] = De[:, -2]

        Dw = np.zeros_like(self.sdf)
        Dw = (padded_sdf[1:-1, 1:-1] - padded_sdf[1:-1, :-2])
        Dw[:, 0] = Dw[:, 1]

        return Dn, Ds, De, Dw # Order: North South East West

    def get_nablas(self) -> tuple[np.ndarray[float], np.ndarray[float]]:
        '''
        Compute quadrature summed directional gradients (smoothing ridges/troughs).
        This helps us enforce upwinding.
        Returns:
            nabla_pos, nabla_neg (tuple): The positive and negative nabla fields (for selection dependent on sgn(V))
        '''
        Dn, Ds, De, Dw = self.get_derivatives()

        nabla_pos = np.zeros_like(self.sdf)
        nabla_neg = np.zeros_like(self.sdf)

        nabla_pos = np.sqrt(np.minimum(Dn, 0)**2 + np.maximum(Ds, 0)**2 + np.minimum(De, 0)**2 + np.maximum(Dw, 0)**2)
        nabla_neg = np.sqrt(np.maximum(Dn, 0)**2 + np.minimum(Ds, 0)**2 + np.maximum(De, 0)**2 + np.minimum(Dw, 0)**2)

        return nabla_pos, nabla_neg # Order: positive, negative

    def step_sdf_numerical_grad(self, iterations : int = 20, dt : float = None):
        '''
        Perform a time step of the SDF update.
        Args:
            iterations (int): The number of update iterations.
            dt (float): The time step size.
        '''
        if dt is None:
            dt = self.dt

        for _ in range(iterations):
            nabla_pos, nabla_neg = self.get_nablas()
            grad = np.maximum(self.V.field, 0) * nabla_pos + np.minimum(self.V.field, 0) * nabla_neg # double check
            self.sdf -= grad * dt
        return

    def step_sdf_analytical_grad(self, iterations : int = 20, dt : float = None):
        '''
        Perform a time step of the SDF update, using the Eikonal assumption : grad = 1.
        If iterations is large, use Level_Set_SDF.step_sdf_numerical_grad to avoid blow up.
        Args:
            iterations (int): The number of update iterations.
            dt (float): The time step size.
        '''
        if dt is None:
            dt = self.dt

        for _ in range(iterations):
            self.sdf -= self.V.field * dt
        return
    

    def get_sdf(self) -> np.ndarray[float]:
        '''
        Get the current SDF (testing function).
        Returns:
            The current SDF (np.ndarray).
        '''
        return self.sdf.copy()

class SDF_MRI(Level_Set_SDF):
    '''
    Daughter class of the Level_Set_SDF class.
    This allows us to turn SDFs into MRI-like data and corresponding segmentation masks.
    '''
    def __init__(self, V : Random_Speed_Field, SDF : np.ndarray[float] = None):
        '''
        Initialize the SDF_MRI class with a seed SDF and speed field.
        Args:
            V (Random_Speed_Field): Speed field
            SDF (np.ndarray or None): Initial signed distance field (must be the same size as V)
        '''
        super().__init__(V, SDF)
        self.N = V.field.shape[0] # We will assume we are always generating square data.

    def activation(self, array : np.ndarray[float], mean_epsilon : float) -> np.ndarray[float]:
        '''
        Apply an activation function to the array to acquire MRI magnitude scan-like behaviour.
        Args:
            array (np.ndarray): The input array to apply the activation function to.
            mean_epsilon (float): A value to determine the boundary layer thickness.
        '''
        epsilon = np.random.normal(mean_epsilon, 0.1)
        return 0.5 * (1 - np.tanh(3 * array / epsilon))

    def cholesky_rbf_1d(self, x : np.ndarray[float], length_scale : float, variance : float) -> np.ndarray[float]:
        '''
        Calculate the Cholesky decomposition of the 1D Radial Basis Function (RBF) kernel.
        Args:
            x (np.ndarray): The input array.
            length_scale (float): The length scale of the RBF.
            variance (float): The variance of the RBF.
        Returns:
            The Cholesky decomposition of the 1D RBF kernel (np.ndarray).
        '''
        # Compute euclidian distances in this direction
        # Note: x[:, None] turns x into a column vector, and x[None, :] turns x into a row vector.
        #       Subtracting one from the other gives the pairwise differences (via broadcasting).
        distances2 = (x[:, None] - x[None, :])**2
        # Compute the kernel in this dimension
        K = variance * np.exp(-0.5 * distances2 / length_scale**2)
        # Add small jitter to the diagonal to make sure K is SPD to numerical precision
        K += np.eye(K.shape[0])*1e-8*variance
        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # If Cholesky fails, add more jitter
            K += np.eye(K.shape[0])*1e-5*variance
            L = np.linalg.cholesky(K)
        return L

    def gaussian_process(self, grid_shape : tuple[int, int] = None, length_scale : float = 1.,
                         variance : float = 1.) -> np.ndarray[float]:
        '''
        Sample random smooth functions from an untrained Gaussian Process. We exploit the separable nature
        of the RBF kernel on a structured grid, i.e. K = Kronecker(K_x, K_y), hence only requiring compute in 1D.
        Args:
            grid_shape (tuple): The shape of the grid to sample on (height, width). If None, uses self.V.shape
            length_scale (float): The length scale of the Gaussian Process (larger -> smoother).
            variance (float): The variance of the Gaussian Process (controls amplitude of function).
        Returns:
            The sampled function (np.ndarray).
        '''
        if grid_shape is None:
            grid_shape = self.V.shape

        # Create coordinate grid
        ny, nx = grid_shape
        y = np.linspace(0, 1, ny)
        x = np.linspace(0, 1, nx)

        Ly = self.cholesky_rbf_1d(y, length_scale=length_scale, variance=variance)
        Lx = self.cholesky_rbf_1d(x, length_scale=length_scale, variance=variance)

        # Sample from standard normal and transform
        # Generate a random latent vector
        rand_gen = np.random.default_rng()
        z = rand_gen.standard_normal((nx, ny))
        samples = Lx @ z @ Ly.T
        return samples
    
    def add_noise(self, arr : np.ndarray[float], noise_level : float = 1.):
        '''
        Add fine Gaussian noise and smooth noise (sampled Gaussian Process) to the SDF.
        Args:
            arr (np.ndarray): The input array to add noise to.
            noise_level (float): The level of noise to add (order-of-magnitude).
        '''
        white_noise = np.random.normal(0, noise_level/5, arr.shape)
        # Generate smooth noise using Gaussian Process
        smooth_noise = self.gaussian_process(grid_shape = arr.shape, length_scale = self.N / 1000, variance = noise_level)
        arr += white_noise + smooth_noise
        return

    def return_mask_magn_pair(self) -> tuple[np.ndarray[float], np.ndarray[float]]:
        '''
        Get the mask and magnitude pair from the SDF.
        Returns:
            A tuple containing the mask and magnitude arrays (tuple).
        '''
        mask = np.where(self.sdf.copy() < 0, 1, 0)
        magn = self.activation(self.sdf.copy(), mean_epsilon = 0.15 * self.N) * 15 # Magnitude ~ 15
        self.add_noise(magn, noise_level = 1)  # Add noise to the magnitude ~ 1
        return mask, magn

class SDF_MRI_Circle(SDF_MRI):
    '''
    Daughter class of the SDF_MRI class.
    Initialises seed SDF for a circle.
    '''
    def __init__(self, V : Random_Speed_Field, r : float = 20, center_var : float = 0.1):
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
            r (float): The circle radius.
            V (Random_Speed_Field): The speed field.
            center_var (float): The variance for center position sampling.
        '''
        super().__init__(V, SDF = None)
        self.r = r
        self.center_var = center_var
        # Set up seed SDF
        # \rho - R for analytical SDF of a circle
        # Randomise center location for location invariance
        center = np.random.multivariate_normal(mean=np.zeros(2), cov=np.eye(2) * center_var) * V.shape // 2 + np.array(V.shape) // 2
        vec = np.stack(np.indices(V.shape), axis = -1)
        self.sdf = np.linalg.norm(vec - center, axis=-1) - r  # Ensure SDF is non-negative outside the circle
        #self.sdf = np.sqrt((vec - center) ** 2).sum(axis=-1) - r

class SDF_MRI_Tube(SDF_MRI):
    '''
    Daughter class of the SDF_MRI class.
    Initialises seed SDF for a tube.
    '''
    def __init__(self, V : Random_Speed_Field, r : float = 10., dir : tuple[float, float] = None, point_var : float = 0.1, smoothed : bool = False):
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
            r (float): The tube radius.
            dir (tuple): The direction vector of the tube. If None, a random direction will be used.
            point_var (float): The variance for point position sampling (to pin the tube).
            smoothed (bool): Modifies the initial SDF to be smoothed.
        '''
        super().__init__(V, SDF = None)
        self.r = r
        self.point_var = point_var
        # Set up seed SDF
        # (r - p).n - t for analytical SDF of a tube
        # Find normal to dir
        if dir is not None:
            n = np.array([-dir[1], dir[0]])
            n /= np.linalg.norm(n)  # Normalize the normal vector
            self.dir = dir / np.linalg.norm(dir)  # Normalize the direction vector
        else:
            angle = np.random.uniform(0, 2 * np.pi)
            n = np.array([np.cos(angle), np.sin(angle)])
            self.dir = np.array([-n[1], n[0]])  # Perpendicular vector to n
        point = np.random.multivariate_normal(mean=np.zeros(2), cov=np.eye(2) * point_var) * V.shape // 2 + np.array(V.shape) // 2
        vec = np.stack(np.indices(V.shape), axis=-1)
        # Compute the signed distance function
        if smoothed:
            self.sdf = np.sqrt((np.dot(vec - point, n))**2 + 2) - r  # SDF is the distance to the tube
        else:
            self.sdf = np.abs(np.dot(vec - point, n)) - r  # SDF is the distance to the tube