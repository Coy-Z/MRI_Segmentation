import torch
import torch.nn as nn
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import Transform
import numpy as np

class ToTensor(Transform):
    '''
    Custom transform for transforming numpy arrays to PyTorch tensors.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, array : np.ndarray) -> torch.Tensor:
        '''
        Args:
            array (np.ndarray): Input numpy array.

        Returns:
            torch.Tensor: Output PyTorch tensor of the same shape and values as input.
        '''
        return torch.from_numpy(array)

class Resize(Transform):
    '''
    Custom transform for resizing PyTorch tensors.
    '''
    def __init__(self, dims : int, size : int | tuple, interpolation: str = 'nearest'):
        '''
        Args:
            size (tuple): The desired output size X, (H, W) or (D, H, W), depending on dimension.
            dims (int): The number of dimensions of the input tensor (2 or 3).
            interpolation (str): The interpolation method to use, bilinear for scans or nearest for masks.

        N.B. The input tensor have shape 2D (N, H, W) or 3D (D, H, W).
             We add redundancy in dimension inference, because nn.functional.interpolate is flexible with size input.
        '''
        assert dims in [2, 3], "Invalid dimensions. Only 2D and 3D tensors are supported."
        super().__init__()
        self.size = size
        self.dims = dims
        self.interpolation = interpolation

    def forward(self, tensor : torch.Tensor) -> torch.Tensor:
        '''
        Args:
            tensor (torch.Tensor): The input tensor to resize of shape 2D (N, H, W) or 3D (D, H, W)

        Returns:
            torch.Tensor: The resized tensor 2D (N, self.size) or 3D (self.size)
        '''
        if self.dims == 2:
            result = nn.functional.interpolate(tensor.unsqueeze(0), size = self.size, mode = self.interpolation).squeeze(0)
        elif self.dims == 3:
            # For 3D tensors, we need to specify the mode as 'trilinear' or 'nearest'
            result = nn.functional.interpolate(tensor.unsqueeze(0).unsqueeze(0), size = self.size, mode = self.interpolation).squeeze(0).squeeze(0)
        else:
            raise ValueError("Unsupported tensor dimensions. Only 2D and 3D tensors are supported.")
        return result

def gaussian_kernel_2d(kernel_size : int, sigma : float, channels : int):
    """
    Generate a 2D Gaussian kernel.
    Args:
        kernel_size (int): Size of the kernel.
        sigma (float): Standard deviation of the Gaussian.
        channels (int): Number of channels.

    Returns:
        torch.Tensor: The generated 2D Gaussian kernel.
    """
    # Coordinate grid
    ax = torch.arange(kernel_size) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()  # normalize

    # Shape (out_channels, in_channels, D, H, W)
    kernel = kernel.expand(channels, channels, *kernel.shape)
    return kernel

def gaussian_kernel_3d(kernel_size : int, sigma : float, channels : int):
    """
    Generate a 3D Gaussian kernel.
    Args:
        kernel_size (int): Size of the kernel.
        sigma (float): Standard deviation of the Gaussian.
        channels (int): Number of channels.

    Returns:
        torch.Tensor: The generated 3D Gaussian kernel.
    """
    # Coordinate grid
    ax = torch.arange(kernel_size) - kernel_size // 2
    xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()  # normalize

    # Shape (out_channels, in_channels, D, H, W)
    kernel = kernel.expand(channels, channels, *kernel.shape)
    return kernel

class GaussianBlur(Transform):
    '''
    Custom transform for applying Gaussian blur to PyTorch tensors.
    '''
    def __init__(self, dims : int = 3, channels : int = 1, kernel_size : int = 5, sigma : float = 0.1):
        '''
        Args:
            dims (int): Number of dimensions (2 or 3).
            channels (int): Number of input channels.
            kernel_size (int): Size of the Gaussian kernel.
            sigma (float): Standard deviation of the Gaussian kernel.
        '''
        super().__init__()
        self.dims = dims
        if dims == 2:
            kernel = gaussian_kernel_2d(kernel_size, sigma, channels)
            self.conv = nn.Conv2d(in_channels = channels, out_channels = channels,
                                  kernel_size = kernel_size, padding = kernel_size // 2,
                                  groups = channels, bias = False)
        elif dims == 3:
            kernel = gaussian_kernel_3d(kernel_size, sigma, channels)
            self.conv = nn.Conv3d(in_channels = channels, out_channels = channels,
                                  kernel_size = kernel_size, padding = kernel_size // 2,
                                  groups = channels, bias = False)
        else:
            raise ValueError("Unsupported dimensions. Only 2D and 3D are supported.")
        
        self.conv.weight.data = kernel
        self.conv.weight.requires_grad_(False)  # Freeze weights
        
    def forward(self, tensor : torch.Tensor) -> torch.Tensor:
        '''
        Args:
            tensor (torch.Tensor): Input tensor 2D (N, H, W) or 3D (D, H, W)

        Returns:
            torch.Tensor: Blurred tensor 2D (N, H, W) or 3D (D, H, W)
        '''
        tensor = tensor.unsqueeze(3 - self.dims)
        return self.conv(tensor).squeeze(3 - self.dims)
        
def clip_and_scale(tensor : torch.Tensor, low_clip : float = 1., high_clip : float = 99., epsilon : float = 1e-6) -> torch.Tensor:
    '''
    Normalize a torch tensor by clipping and scaling to [0, 1].

    Args:
        tensor (torch.Tensor): Input image Float32 tensor (any shape).
        low_clip (float): Lower percentile to clip at.
        high_clip (float): Upper percentile to clip at.

    Returns:
        tensor (torch.Tensor): Normalized Float32 tensor scaled to [0, 1].
    '''
    # Clip
    flat = tensor.flatten()
    lower = torch.quantile(flat, low_clip / 100)
    upper = torch.quantile(flat, high_clip / 100)
    tensor = torch.clamp(tensor, min = lower, max = upper)

    # Scale to [0, 1]
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + epsilon)
    return tensor

def clip_and_scale_slices(tensor : torch.Tensor, low_clip : float = 1., high_clip : float = 99., epsilon : float = 1e-6) -> torch.Tensor:
    '''
    Normalize a torch tensor by clipping and scaling each slice to [0, 1].

    Args:
        tensor (torch.Tensor): Input image Float32 tensor (any shape).
        low_clip (float): Lower percentile to clip at.
        high_clip (float): Upper percentile to clip at.

    Returns:
        tensor (torch.Tensor): Normalized Float32 tensor, where each slice is scaled to [0, 1].
    '''
    # Flatten each slice
    flat = tensor.flatten(1)

    # Find lower and upper percentiles
    lower = torch.quantile(flat, low_clip / 100, dim = 1, keepdim = True)
    upper = torch.quantile(flat, high_clip / 100, dim = 1, keepdim = True)

    # Clip per slice
    flat = torch.clamp(flat, min = lower, max = upper)

    # Reshape
    clipped = flat.view_as(tensor)

    # Scale per slice
    min_vals = flat.min(dim = 1, keepdim = True)[0].view([-1] + [1]*(tensor.dim()-1))
    max_vals = flat.max(dim = 1, keepdim = True)[0].view([-1] + [1]*(tensor.dim()-1))
    result = (clipped - min_vals) / (max_vals - min_vals + epsilon)
    return result

class ClipAndScale(Transform):
    '''
    Custom transform to clip and scale a tensor.
    For 2D slices, clips and scales per layer.
    For 3D volumes, clips and scales the entire volume.
    '''
    def __init__(self, dims : int = 3, low_clip : float = 1., high_clip : float = 99., epsilon : float = 1e-8):
        '''
        Args:
            dims (int): Number of dimensions (2 or 3).
            low_clip (float): Lower percentile to clip at.
            high_clip (float): Upper percentile to clip at.
            epsilon (float): Small value to avoid division by zero.
        '''
        super().__init__()
        self.dims = dims
        self.low_clip = low_clip
        self.high_clip = high_clip
        self.epsilon = epsilon

    def forward(self, tensor : torch.Tensor) -> torch.Tensor:
        if self.dims == 2:
            return clip_and_scale_slices(tensor, self.low_clip, self.high_clip, self.epsilon)
        else:
            return clip_and_scale(tensor, self.low_clip, self.high_clip, self.epsilon)

class RandomXFlip(Transform):
    '''
    Randomly flips tensor in W-dimension (-1).
    '''
    def __init__(self, p = 0.5):
        '''
        Args:
            p (float): Probability of flipping.
        '''
        super().__init__()
        self.p = p

    def forward(self, tensor : torch.Tensor) -> torch.Tensor:
        '''
        Args:
            tensor (torch.Tensor): Input tensor to randomly flip.

        Returns:
            torch.Tensor: Randomly flipped tensor.
        '''
        if torch.rand(1) < self.p:
            tensor = torch.flip(tensor, dims = [-1])
        return tensor

class RandomYFlip(Transform):
    '''
    Flips tensor in H-dimension (-2).
    '''
    def __init__(self, p = 0.5):
        '''
        Args:
            p (float): Probability of flipping.
        '''
        super().__init__()
        self.p = p

    def forward(self, tensor : torch.Tensor) -> torch.Tensor:
        '''
        Args:
            tensor (torch.Tensor): Input tensor to randomly flip.

        Returns:
            torch.Tensor: Randomly flipped tensor.
        '''
        if torch.rand(1) < self.p:
            tensor = torch.flip(tensor, dims = [-2])
        return tensor

class RandomZFlip(Transform):
    '''
    Flips tensor in D-dimension (-3)
    '''
    def __init__(self, p = 0.5):
        '''
        Args:
            p (float): Probability of flipping.
        '''
        super().__init__()
        self.p = p

    def forward(self, tensor : torch.Tensor) -> torch.Tensor:
        '''
        Args:
            tensor (torch.Tensor): Input tensor to randomly flip.

        Returns:
            torch.Tensor: Randomly flipped tensor.
        '''
        if torch.rand(1) < self.p:
            tensor = torch.flip(tensor, dims = [-3])
        return tensor

class RandomRotation(Transform):
    def __init__(self, degrees : float):
        '''
        Args:
            degrees (float): Maximum rotation angle in degrees. Rotation will be in range [-degrees, degrees].
        '''
        super().__init__()
        self.degrees = degrees

    def forward(self, tensor : torch.Tensor) -> torch.Tensor:
        '''
        Args:
            tensor (torch.Tensor): Input tensor to randomly rotate.

        Returns:
            torch.Tensor: Randomly rotated tensor.
        '''
        if torch.rand(1) > 0.5:
            angle = torch.randint(-self.degrees, self.degrees + 1, (1,)).item()
            tensor = T.functional.rotate(tensor, angle)
        return tensor
    
class GaussianNoise(Transform):
    def __init__(self, mean : float = 0.0, sigma : float = 0.1, clip : bool = False):
        '''
        Args:
            mean (float): Mean of the Gaussian noise.
            sigma (float): Standard deviation of the Gaussian noise.
            clip (bool): Whether to clip the noise to [0, 1].
        '''
        super().__init__()
        self.mean = mean
        self.sigma = sigma
        self.clip = clip

    def forward(self, tensor : torch.Tensor) -> torch.Tensor:
        '''
        Args:
            tensor (torch.Tensor): Input tensor to add Gaussian noise.

        Returns:
            torch.Tensor: Tensor with added Gaussian noise.
        '''
        noise = torch.randn_like(tensor) * self.sigma + self.mean
        if self.clip:
            noise = torch.clamp(noise, 0, 1)
        return tensor + noise