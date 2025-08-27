import numpy as np
import os
import torch
import torch.nn as nn
from torchvision.transforms import v2 as T
from torch.utils.data import Dataset
from typing import Sequence
from utils.custom_transforms import ToTensor, Resize, GaussianBlur, ClipAndScale, RandomXFlip, RandomYFlip, RandomZFlip, RandomRotation

class MRIDataset(Dataset):
    '''
    A custom dataset class for processing .npy 3D MRI density scans.
    '''
    def __init__(self, root : str, phase : str = 'val', dims : int = 2, transform : T.Compose = None,
                 target_transform : T.Compose = None, augment : T.Compose = None):
        '''
        Initialize the MRIDataset daughter class of torch.utils.data.Dataset.

        Args:
            root (str): The string directory of the data.
            phase (str): Indicating whether it is a training or validation dataset.
            dims (int): The number of dimensions for the MRI scans (2 or 3).
            transform (T.Compose): The deterministic component of magn transform (i.e. no random flipping -> validation transform).
            target_transform (T.Compose): The deterministic component of mask transform (i.e. no random flipping -> validation transform).
            augment (T.Compose): The stochastic component of transform (e.g. random horizontal flip -> additional training transform).

        Note: transform and augment are separate because we must merge the scan and mask to ensure consistent augmentation.
              However, interpolation schemes for the two are different.
        '''
        assert dims in [2, 3], "Invalid dimensions. Only 2D and 3D scans are supported."
        super().__init__()
        self.scans = list(sorted(os.listdir(os.path.join(root, "magn")))) # (N * D * H * W)
        self.masks = list(sorted(os.listdir(os.path.join(root, "mask")))) # (N * D * H * W)
        self.root = root
        self.phase = phase
        self.dims = dims
        self.transform = transform
        self.target_transform = target_transform
        self.augment = augment
    
    def __len__(self) -> int:
        '''
        Return the number of scans in the Dataset.

        Returns:
            int: The number of scans in the dataset.
        '''
        return len(self.scans)
    
    def __getitem__(self, idx : int) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Preprocess and return the idx-th scan from the array of scans.

        Args:
            idx (int): The index of the scan we wish to retrieve from the dataset.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the preprocessed scan and mask tensors.
        '''
        scan_path = os.path.join(self.root, "magn", self.scans[idx])
        mask_path = os.path.join(self.root, "mask", self.masks[idx])
        
        # Load data
        scan = np.load(scan_path) # (D * H * W)
        mask = np.load(mask_path) # (D * H * W)
        
        if mask.dtype == bool:
            mask = mask.astype(np.int64)  # False->0, True->1
        
        # Apply transforms
        if self.transform:
            scan = self.transform(scan) # (D * H * W)
        if self.target_transform:
            mask = self.target_transform(mask) # (D * H * W)

        if self.augment is None or self.phase == 'val':
            return scan, mask

        data = torch.cat([scan.unsqueeze(0), mask.unsqueeze(0)], dim=0) # (2 * D * H * W)
        data = self.augment(data) # Need to ensure augment knows whether to work on 2D or 3D data.
        scan = data[0, :, :, :] # (1 * D * H * W)
        mask = data[1, :, :, :] # (1 * D * H * W)
        return scan, mask

class U_Net_Skip_Block(nn.Module):
    '''
    Skip Connections for the U-Net model.
    
    Note: The U-Net architecture at present requires sizes to be powers of 2, to account for correct upsampling.
          This can be tackled by introducing a crop in the skip block.
    '''
    def __init__(self, dims : int, block : nn.Module, in_channels : int, out_channels : int):
        '''
        Initialize the modules that we will be using in this skip block.

        Args:
            dims (int): The number of spatial dimensions (2 or 3).
            block (nn.Module): The block to be skipped.
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.

        Note: padding = 1 means we do not change the spatial dimensions between input and output, which is required for the task at hand.
              The original U-Net paper does not use padding, and instead crops the skip.
        '''
        assert dims in [2, 3], "Invalid spatial dimensions"
        super().__init__()
        self.dims = dims
        self.relu = nn.ReLU(inplace=True)
        if dims == 3:
            # Conv 3x3 modules
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size = 3, padding = 1)
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size = 3, padding = 1)
            self.conv3 = nn.Conv3d(2 * out_channels, out_channels, kernel_size = 3, padding = 1)
            # Downsampling and upsampling modules
            self.maxpool = nn.MaxPool3d(kernel_size = 2, stride = 2)
            self.deconv = nn.ConvTranspose3d(2 * out_channels, out_channels, kernel_size = 2, stride = 2)
        else:
            # Conv 3x3 modules
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
            self.conv3 = nn.Conv2d(2 * out_channels, out_channels, kernel_size = 3, padding = 1)
            # Downsampling and upsampling modules
            self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
            self.deconv = nn.ConvTranspose2d(2 * out_channels, out_channels, kernel_size = 2, stride = 2)
        # Block that is to be skipped over
        self.block = block        

    def forward(self, input : torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the skip block.

        Args:
            input (torch.Tensor): Input tensor of shape 2D (B*, D, in_channels, H, W) or 3D (B*, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape 2D (B*, D, out_channels, H, W) or 3D (B*, out_channels, D, H, W).

        * B is optional.
        ** D is depth in 3D case and slice/frame in 2D case.
        '''
        # Level n
        x1 = self.conv1(input)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.relu(x1)
        x2 = self.maxpool(x1)

        # Level n + 1
        x2 = self.block(x2)
        x2 = self.deconv(x2)

        # Level n
        x3 = torch.cat([x1, x2], dim = -(self.dims + 1))  # Concatenate along channel dimension
        x3 = self.conv3(x3)
        x3 = self.relu(x3)
        x3 = self.conv2(x3)
        x3 = self.relu(x3)
        return x3

class U_Net(nn.Module):
    '''
    A U-Net model for medical image segmentation.
    '''
    def __init__(self, dims : int, num_classes : int):
        '''
        Initialise the full U-Net model.
        Having constructed the U_Net_Skip_Block, we can simply nest them as below.

        Args:
            dims (int): The number of spatial dimensions (2 or 3).
            num_classes (int): The number of output classes for segmentation.
        '''
        assert dims in [2, 3], "Invalid spatial dimensions"
        super().__init__()
        if dims == 3:
            self.level_4 = nn.Sequential(
                nn.Conv3d(512, 1024, kernel_size = 3, padding = 1),
                nn.ReLU(inplace = True),
                nn.Conv3d(1024, 1024, kernel_size = 3, padding = 1),
                nn.ReLU(inplace = True)
            )
        else:
            self.level_4 = nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size = 3, padding = 1),
                nn.ReLU(inplace = True),
                nn.Conv2d(1024, 1024, kernel_size = 3, padding = 1),
                nn.ReLU(inplace = True)
            )
        
        self.level_3 = U_Net_Skip_Block(dims, self.level_4, 256, 512)
        self.level_2 = U_Net_Skip_Block(dims, self.level_3, 128, 256)
        self.level_1 = U_Net_Skip_Block(dims, self.level_2, 64, 128)
        self.level_0 = U_Net_Skip_Block(dims, self.level_1, 1, 64)
        if dims == 3:
            self.network = nn.Sequential(
                self.level_0,
                nn.Conv3d(64, num_classes, kernel_size = 1)
            )
        else:
            self.network = nn.Sequential(
                self.level_0,
                nn.Conv2d(64, num_classes, kernel_size = 1)
            )
    
    def forward(self, input : torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the model.

        Args:
            input (torch.Tensor): Input tensor of shape (B*, 1, D**, H, W).
        
        Returns:
            torch.Tensor: Output tensor of shape (B*, 1, D**, H, W).
        
        * B is optional.
        ** D is only present in 3D case.
        '''
        return self.network(input)
    
    def _init_weights(self):
        '''
        Initialize weights using conservative initialization for stability.
        Uses Xavier/Glorot initialization for transpose convolutions and downscaled He init for regular convolutions.
        '''
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                # Use small gain for He initialization to prevent gradient explosion
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu', a=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.ConvTranspose2d, nn.ConvTranspose3d)):
                # Use Xavier initialization for transpose convolutions for stability
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Combined_Loss(nn.Module):
    '''
    A custom loss function class for a combined cross-entropy and DICE loss.
    '''
    def __init__(self, device, dims : int = 3, alpha : float = 1., beta : float = 0.7, gamma : float = 0.75,
                 epsilon : float = 1e-8, ce_weights : Sequence[float] = [0.1, 0.9]):
        '''
        Initialise the Combined_Loss daughter class of torch.nn.Module.

        Args:
            device (torch.device): The device for computation.
            alpha (float): Relative weighting of Cross-Entropy and Dice losses (N.B. only relative magnitude matters, i.e. alpha = 0 -> Cross-Entropy loss dominates).
            beta (float): Relative weighting of false positives and false negatives in Tversky loss (i.e. beta = 0 -> no false negatives, beta = 1 -> no false positives).
            gamma (float): The Focal loss exponent.
            epsilon (float): The smoothing factor.
            ce_weights (Sequence[float]): The relative weighting of classes within the Cross-Entropy loss.
        '''
        assert dims in [2, 3], 'Invalid spatial dimensions'
        super().__init__()
        self.dims = dims
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.CELoss = nn.CrossEntropyLoss(weight = torch.tensor(ce_weights).to(device))

    def forward(self, output : torch.Tensor, target : torch.Tensor) -> float:
        '''
        Calculates the combined cross-entropy and DICE loss, i.e. forward pass of combined loss function.
        
        Args:
            output (torch.Tensor): Predicted mask logits Float32 tensor 2D (D * 2 * H * W) or 3D (1 * 2 * D * H * W).
            target (torch.Tensor): Ground truth binary (0,1) mask Int64 tensor 2D (D * H * W) or 3D (1 * D * H * W).
        
        Returns:
            float: The combined loss.
        '''
        CE = self.CELoss(output, target)
        DICE = self.DiceLoss(output, target, self.epsilon)
        FOC_TVSKY = self.FocalTverskyLoss(output, target, self.epsilon)
        return FOC_TVSKY
        return DICE
        return self.alpha * FOC_TVSKY + CE
    
    def DiceLoss(self, output : torch.Tensor, target : torch.Tensor, epsilon : float = 1e-8) -> float:
        '''
        Calculate the Dice loss of a predicted mask with respect to the ground truth.

        Args:
            output (torch.Tensor): Predicted mask logits Float32 tensor 2D (D * 2 * H * W) or 3D (1 * 2 * D * H * W).
            target (torch.Tensor): Ground truth binary (0,1) mask Int64 tensor 2D (D * H * W) or 3D (1 * D * H * W).
            epsilon (float): The smoothing factor.
        
        Returns:
            float: The Dice loss (1 - Dice coefficient). The negation allows for gradient descent.
        '''
        # Softmax the logits to probabilities
        probs = torch.softmax(output, dim = 1)

        # Note the one-hot is actually redundant but can be useful if generalising to multiple classes.
        # Create one-hot encoding
        target_onehot = torch.zeros_like(output)
        target_onehot.scatter_(1, target.unsqueeze(1), 1)
        
        # Focus on foreground class (class 1)
        probs_fg = probs[:, 1].squeeze()
        target_fg = target_onehot[:, 1].squeeze()
        
        # Calculate intersection and union in one pass
        sum_dims = (0, 1, 2) if self.dims == 3 else (1, 2) 
        intersection = (probs_fg * target_fg).sum(dim = sum_dims)
        probs_sum = probs_fg.sum(dim = sum_dims)
        target_sum = target_fg.sum(dim = sum_dims)

        dice_coeff = (2 * intersection + epsilon) / (probs_sum + target_sum + epsilon)
        dice_loss = 1 - dice_coeff
        return dice_loss.mean() # Average over batch/depth

    def FocalTverskyLoss(self, output : torch.Tensor, target : torch.Tensor, epsilon : float = 1e-8) -> float:
        '''
        Calculate the Focal Tversky loss of a predicted mask with respect to the ground truth.

        Args:
            output (torch.Tensor): Predicted mask logits Float32 tensor 2D (D * 2 * H * W) or 3D (1 * 2 * D * H * W).
            target (torch.Tensor): Ground truth binary (0,1) mask Int64 tensor 2D (D * H * W) or 3D (1 * D * H * W).
            epsilon (float): The smoothing factor.
        
        Returns:
            float: The Focal Tversky loss.
        '''
        # Hyperparameters
        alpha = 1 - self.beta
        beta = self.beta
        gamma = self.gamma
        
        # Softmax the logits to probabilities
        probs = torch.softmax(output, dim = 1) # 2D (D * 2 * H * W) or 3D (1 * 2 * D * H * W)
        
        # Note the one-hot is actually redundant but can be useful if generalising to multiple classes.
        # Create one-hot encoding
        target_onehot = torch.zeros_like(output)
        target_onehot.scatter_(1, target.unsqueeze(1), 1) # 2D (D * 2 * H * W) or 3D (1 * 2 * D * H * W)

        # Focus on foreground class (class 1)
        probs_fg = probs[:, 1].squeeze() # (D * H * W)
        target_fg = target_onehot[:, 1].squeeze() # (D * H * W)

        # Calculate TP, FP, FN and Loss function
        sum_dims = (0, 1, 2) if self.dims == 3 else (1, 2) 
        TP = (probs_fg * target_fg).sum(dim = sum_dims)
        FP = (probs_fg * (1 - target_fg)).sum(dim = sum_dims)
        FN = ((1 - probs_fg) * target_fg).sum(dim = sum_dims)

        tversky_coeff = (TP + epsilon) / (TP + alpha * FP + beta * FN + epsilon)
        focal_tversky_loss = (1 - tversky_coeff) ** gamma
        return focal_tversky_loss.mean() # Average over batch/depth

def sum_IoU(pred_mask : torch.Tensor, true_mask : torch.Tensor) -> float:
    '''
    Calculate the Intersection over Union (IoU) value between the predicted mask and ground truth.

    Args:
        pred_mask (torch.Tensor): A binary (0,1) predicted mask Int64 tensor (any shape).
        true_mask (torch.Tensor): A binary (0,1) ground truth mask Int64 tensor (same shape as pred_mask)
    
    Returns:
        float: The IoU score of the two mask tensors.
    '''
    pred_bool = pred_mask.squeeze().bool()
    true_bool = true_mask.squeeze().bool()

    intersection = (pred_bool & true_bool).sum().item()
    union = (pred_bool | true_bool).sum().item()

    if union == 0: # Handle empty masks
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / union

def get_model_instance_unet(num_classes : int, device : str = 'cpu', dims : int = 3, trained : bool = False) -> U_Net:
    '''
    Load an instance of the model of choice, with pre-trained weights.

    Args:
        num_classes (int): The number of output classes. Here, we use two -> 0. Background | 1. Blood vessel
        device (str): The device to run the model on ('cpu' or 'cuda').
        dims (int): The number of dimensions for the input data (2 or 3).
        trained (bool): A boolean depicting whether the model has been locally trained or not, i.e. whether to load fine-tuned or default weights.

    Returns:
        U_Net: The model with required weights.
    '''
    assert dims in [2, 3], "Invalid dimensions. Only 2D and 3D data is supported."
    model = U_Net(dims = dims, num_classes = num_classes)

    if trained: # If the model has been locally trained, load the fine-tuned weights
        model.load_state_dict(torch.load(f'{dims}D_model_params.pth', weights_only = True))
    else:
        # Initialize weights more conservatively for better training stability
        model._init_weights()

    return model.to(device) # Move the model to the specified device

def _get_transform(data : str = 'target', phase : str = 'train') -> T.Compose:
    '''
    DEPRECATED
    Get the appropriate transform for the input data.

    Args:
        phase (str): The phase of the dataset, either 'train' or 'val'.
        data (str): The type of input data, either 'input' for scans or 'target' for masks.

    Returns:
        T.Compose: The composed transform for the specified input type.
    '''
    # Note: BILINEAR for images (smooth), NEAREST for masks (preserve labels)
    if data == 'target':
        interpolation = T.InterpolationMode.NEAREST
        # For masks: don't scale, keep as integers for proper class labels
        return T.Compose([
            T.ToImage(),
            T.ToDtype(torch.int64, scale = False),  # Keep integer labels, no scaling
            T.Resize(size = (64, 64), interpolation = interpolation),
        ])
    else: 
        interpolation = T.InterpolationMode.BILINEAR
        if phase == 'train':
            return T.Compose([
                T.ToImage(),
                T.ToDtype(torch.float32, scale = True),
                T.Resize(size = (64, 64), interpolation = interpolation),
                #T.RandomResizedCrop(size = (50, 50), scale = (0.5, 1.5), interpolation = interpolation),  # vary size
                T.GaussianBlur(kernel_size = 5, sigma = 0.1),
                ClipAndScale()
            ])
        else:  # validation phase
            return T.Compose([
                T.ToImage(),
                T.ToDtype(torch.float32, scale = True),
                T.Resize(size = (64, 64), interpolation = interpolation),
                T.GaussianBlur(kernel_size = 5, sigma = 0.1),
                ClipAndScale()
            ])

def get_transform(data : str = 'target', dims : int = 3) -> T.Compose:
    '''
    Get the appropriate transform for the input data.
    Must apply to either 3D (D, H, W, C) or 2D (N, H, W, C).

    Args:
        data (str): The type of input data, either 'input' for scans or 'target' for masks.
        dims (int): The number of dimensions for the data (2 or 3).

    Returns:
        T.Compose: The composed transform for the specified input type.
    '''
    assert dims in [2, 3], "Invalid dimensions. Only 2D and 3D data is supported."
    if dims == 3:
        size = (64, 64, 64)
    else:
        size = (64, 64)
    if data == 'target':
        return T.Compose([
            ToTensor(),
            T.ToDtype(torch.float32, scale = False),
            Resize(dims = dims, size = size, interpolation = 'nearest'),
            T.ToDtype(torch.int64, scale = False)  # Convert to int64 after resizing
        ])
    else:
        interpolation = 'bilinear' if dims == 2 else 'trilinear'
        return T.Compose([
            ToTensor(),
            T.ToDtype(torch.float32, scale = True),
            GaussianBlur(dims = dims, kernel_size = 5, sigma = 0.1),
            Resize(dims = dims, size = size, interpolation = interpolation),
            ClipAndScale(dims = dims, low_clip = 1., high_clip = 99., epsilon = 1e-8)
        ])

def get_augment(dims : int) -> T.Compose:
    """
    Get the appropriate augmentations for the input data.

    Args:
        dims (int): The number of dimensions for the data (2 or 3).

    Returns:
        T.Compose: The composed augmentations for the specified dimensions.
    """
    assert dims in [2, 3], "Invalid dimensions. Only 2D and 3D data is supported."
    if dims == 3:
        return T.Compose([
            RandomXFlip(p = 0.5),
            RandomYFlip(p = 0.5),
            RandomZFlip(p = 0.5),
            RandomRotation(degrees = 15),
        ])
    else:
        return T.Compose([
            RandomXFlip(p = 0.5),
            RandomYFlip(p = 0.5),
            RandomRotation(degrees = 15),
        ])

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized 3D scans.
    Returns the first item since we can't batch variable-sized 3D volumes
    """
    return batch[0]