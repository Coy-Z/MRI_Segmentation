import torch
import torch.nn as nn
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import Transform
import numpy as np
import matplotlib.pyplot as plt

def grayscale_to_rgb(scan : np.ndarray[float], cmap : str = 'inferno') -> np.ndarray[float]:
    '''
    Colours a greyscale intensity plot.

    Args:
        scan (np.ndarray): Input greyscale scan array (D * H * W).
        cmap (string): Choice of colormap, out of the matplotlib strings - grey, bone, viridis, plasma, inferno etc...

    Returns:
        scan (np.ndarray): Output RGB scan array (D * H * W * 3).
    '''
    # Normalize to [0, 1] range first
    scan_norm = (scan - scan.min()) / (scan.max() - scan.min() + 1e-8)

    if cmap == 'inferno' or cmap == 'viridis':
        # Simple fast approximation - can be replaced with lookup table for production
        scan_rgb = np.stack([scan_norm, scan_norm, scan_norm], axis=-1) * 255
    else:
        # Fallback to matplotlib for other colormaps
        cmap_func = plt.get_cmap(cmap)
        scan_rgb = cmap_func(scan_norm) * 255 # (D * H * W * 4)
        scan_rgb = scan_rgb[..., :3]  # Remove alpha channel (D * H * W * 3)
    
    return scan_rgb

class ToTensor(Transform):
    '''
    Custom transform for transforming numpy arrays to PyTorch tensors.
    '''
    def __init__(self, dims : int = 3):
        super().__init__()
        self.dims = dims

    def forward(self, array : np.ndarray) -> torch.Tensor:
        '''
        Args:
            array (np.ndarray): Input numpy array.

        Returns:
            torch.Tensor: Output PyTorch tensor.
        '''
        return torch.from_numpy(array)

class Resize(Transform):
    '''
    Custom transform for resizing PyTorch tensors.
    '''
    def __init__(self, size : int | tuple, dims : int = 3, interpolation: str = 'nearest'):
        '''
        Args:
            size (tuple): The desired output size X, (H, W) or (D, H, W), depending on dimension.
            dims (int): The number of dimensions of the input tensor (2 or 3).
            interpolation (str): The interpolation method to use, bilinear for scans or nearest for masks.
        N.B. The input tensor must be in the format (C, D, H, W) for 3D or (N, C, H, W) for 2D.
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
            tensor (torch.Tensor): The input tensor to resize of shape (D, H, W) or (N, H, W)

        Returns:
            torch.Tensor: The resized tensor 2D (N, self.size) or 3D (self.size)
        '''
        if self.dims == 2:
            result = nn.functional.interpolate(tensor.unsqueeze(0), size=self.size, mode=self.interpolation).squeeze(0)
        elif self.dims == 3:
            result = nn.functional.interpolate(tensor.unsqueeze(0).unsqueeze(0), size=self.size, mode=self.interpolation).squeeze(0).squeeze(0)
        else:
            raise ValueError("Unsupported tensor dimensions. Only 2D and 3D tensors are supported.")
        return result

def gaussian_kernel_2d(kernel_size: int, sigma: float):
    """Creates a 2D Gaussian kernel."""
    # coordinate grid
    ax = torch.arange(kernel_size) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()  # normalize
    return kernel

def gaussian_kernel_3d(kernel_size: int, sigma: float):
    """Creates a 3D Gaussian kernel."""
    # coordinate grid
    ax = torch.arange(kernel_size) - kernel_size // 2
    xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()  # normalize
    return kernel

class GaussianBlur(Transform):
    '''
    Custom transform for applying Gaussian blur to PyTorch tensors.
    '''
    def __init__(self, dims : int = 3, kernel_size : int = 5, sigma : float = 0.1):
        assert dims in [2, 3], 'Invalid dimensions. Only 2D and 3D are supported.'
        super().__init__()
        self.dims = dims
        if dims == 2:
            kernel = gaussian_kernel_2d(kernel_size, sigma)
            self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size,
                                  padding=kernel_size//2, groups=1, bias=False)
        elif dims == 3:
            kernel = gaussian_kernel_3d(kernel_size, sigma)
            self.conv = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=kernel_size,
                                  padding=kernel_size//2, groups=1, bias=False)
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
        return self.conv(tensor.unsqueeze(0)).squeeze(0)
        
def clip_and_scale(tensor : torch.Tensor, low_clip : float = 1., high_clip : float = 99., epsilon : float = 1e-6) -> torch.Tensor:
    '''
    Normalize a torch tensor image by clipping and scaling to [0, 1].

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

class ClipAndScale(Transform):
    def __init__(self, dims : int = 3, low_clip : float = 1., high_clip : float = 99., epsilon : float = 1e-8):
        super().__init__()
        self.dims = dims
        self.low_clip = low_clip
        self.high_clip = high_clip
        self.epsilon = epsilon

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.dims == 2:
            N, C, _, _ = tensor.shape
            flat = tensor.view(N, C, -1)  # flatten spatial dimensions

            # Compute lower and upper percentiles per sample and channel
            lower = torch.quantile(flat, self.low_clip / 100, dim=2, keepdim=True)
            upper = torch.quantile(flat, self.high_clip / 100, dim=2, keepdim=True)

            # Clip using broadcasting
            tensor = torch.max(torch.min(tensor, upper.view(N, C, 1, 1)), lower.view(N, C, 1, 1))

            # Compute min and max per sample and channel
            min_val = flat.min(dim=2, keepdim=True)[0].view(N, C, 1, 1)
            max_val = flat.max(dim=2, keepdim=True)[0].view(N, C, 1, 1)

            # Scale to [0,1]
            tensor = (tensor - min_val) / (max_val - min_val + self.epsilon)

            return tensor
        else:
            return clip_and_scale(tensor, self.low_clip, self.high_clip, self.epsilon)