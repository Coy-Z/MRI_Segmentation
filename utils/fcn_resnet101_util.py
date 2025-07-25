import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
from typing import Sequence


class MRIDataset(Dataset):
    '''
    A custom dataset class for processing .npy 3D MRI density scans.
    '''
    def __init__(self, root : str, phase : str = 'val', transform = None, target_transform = None, augment = None):
        '''
        Initialise the MRIDataset daughter class of torch.utils.data.Dataset.

        Args:
            root (str): The string directory of the data.
            phase (str): Indicating whether it is a training or validation dataset.
            transform: The deterministic component of transform (i.e. no random flipping -> validation transform).
            augment: The stochastic component of transform (e.g. random horizontal flip -> additional training transform).
            
        N.B. transform and augment are separate because we must merge the scan and mask to ensure consistent augmentation.
        '''
        # Currently incomplete, depends on how data is laid out. ----
        super().__init__()
        #self.scans = np.array([np.load(f'{root}/magn.npy')]) # (N * D * H * W)
        #self.masks = np.array([np.load(f'{root}/mask.npy')]) # (N * D * H * W)
        self.scans = list(sorted(os.listdir(os.path.join(root, "magn")))) # (N * D * H * W)
        self.masks = list(sorted(os.listdir(os.path.join(root, "mask")))) # (N * D * H * W)
        self.root = root
        self.phase = phase
        self.transform = transform
        self.target_transform = target_transform
        self.augment = augment
    
    def __len__(self):
        '''
        Return the number of scans in the Dataset.
        '''
        return len(self.scans)
    
    def __getitem__(self, idx : int) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Preprocess and return the idx-th scan from the array of scans.

        Args:
            idx (int): The index of the scan we wish to retrieve from the dataset.
        
        Returns:
            scan (torch.Tensor): The pre-processed RGB scan Float32 tensor (D * 3 * H * W).
            mask3d (torch.Tensor): The pre-processed binary mask Int64 tensor (D * H * W).
        '''
        scan_path = os.path.join(self.root, "magn", self.scans[idx])
        mask_path = os.path.join(self.root, "mask", self.masks[idx])
        scan = grayscale_to_rgb(np.load(scan_path)) # (D * H * W * 3)
        mask3d = np.load(mask_path)[..., np.newaxis] # (D * H * W * 1)

        if self.transform:
            scan = torch.stack([self.transform(slice) for slice in scan]) # (D * 3 * H * W)
        if self.target_transform:
            mask3d = torch.stack([self.target_transform(mask) for mask in mask3d]) # (D * 1 * H * W)
        if self.augment and self.phase == 'train':
            data = torch.cat([scan, mask3d], 1) # (D * 4 * H * W)
            data = self.augment(data)

            scan = data[:, :3, :, :] # (D * 3 * H * W)
            mask3d = data[:, 3:, :, :] # (D * 1 * H * W)
        return scan, mask3d.squeeze(1)
    
    #def __getitems__(self, idxs : list[int]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return
    
    def grayscale_to_rgb(self, scan: np.ndarray[float], cmap : str = 'inferno') -> np.ndarray[float]:
        '''
        Colours a greyscale intensity plot.

        Args:
            scan (np.ndarray): Input greyscale scan array (D * H * W).
            cmap (string): Choice of colormap, out of the matplotlib strings - grey, bone, viridis, plasma, inferno etc...
        
        Returns:
            scan (np.ndarray): Output RGB scan array (D * 3 * H * W).
        '''
        # Convert greyscale to cmap and multiply by 255
        cmap = plt.get_cmap(cmap)
        scan = cmap(scan/scan.max())*255 # (D * H * W * 4)
        # Convert RGBA to RGB
        scan = np.delete(scan, 3, axis = 3) # (D * H * W * 3)
        return scan
    
class Combined_Loss(nn.Module): # Need to complete ----
    '''
    A custom loss function class for a combined cross-entropy and DICE loss.
    '''
    def __init__(self, device, alpha : float = 1., beta : float = 0.7, gamma : float = 0.75,
                 epsilon : float = 1e-8, ce_weights : Sequence[float] = [0.5, 0.5]):
        '''
        Initialise the CE_Dice_Loss daughter class of torch.nn.Module.

        Args:
            alpha (float): Relative weighting of Cross-Entropy and Dice losses (N.B. only relative magnitude matters).
            beta (float): Relative weighting of false positives and false negatives in Tversky loss.
            gamma (float): The Focal loss exponent.
            epsilon (float): The smoothing factor.
            ce_weights (iterable): The relative weighting of classes within the Cross-Entropy loss.
        '''
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.CELoss = nn.CrossEntropyLoss(weight = torch.tensor(ce_weights).to(device))

    def forward(self, output : torch.Tensor, target : torch.Tensor) -> float:
        '''
        Calculates the combined cross-entropy and DICE loss, i.e. forward pass of combined loss function.
        
        Args:
            output (torch.Tensor): Predicted mask logits Float32 tensor (D * 2 * H * W).
            target (torch.Tensor): Ground truth binary (0,1) mask Int64 tensor (D * H * W).
        
        Returns:
            loss (float): The combined loss.
        '''
        CE = self.CELoss(output, target)
        DICE = self.DiceLoss(output, target, self.epsilon)
        FOC_TVSKY = self.FocalTverskyLoss(output, target, self.epsilon)
        return self.alpha * FOC_TVSKY + CE
    
    def DiceLoss(self, output : torch.Tensor, target : torch.Tensor, epsilon : float = 1e-8) -> float:
        '''
        Calculate the Dice loss of a predicted mask with respect to the ground truth.

        Args:
            output (torch.Tensor): Predicted mask logits Float32 tensor (D * 2 * H * W).
            target (torch.Tensor): Ground truth binary (0,1) mask Int64 tensor (D * H * W).
            epsilon (float): The smoothing factor.
        
        Returns:
            dice_loss (float): The Dice loss (1 - Dice coefficient). The negation allows for gradient descent.
        '''
        # Softmax the logits to probabilities
        probs = torch.softmax(output, dim = 1)
        # Excite the target tensor to shape (D * 2 * H * W)
        target_onehot = torch.nn.functional.one_hot(target, num_classes = 2).permute(0, 3, 1, 2).float()
        # Focus on foreground class (class 1)
        probs_fg = probs[:, 1, :, :]
        target_fg = target_onehot[:, 1, :, :]
        intersection = (probs_fg * target_fg).sum(dim = (1, 2))
        dice_coeff = (2 * intersection + epsilon ) / (probs_fg.sum(dim = (1, 2)) + target_fg.sum(dim = (1, 2)) + epsilon)
        dice_loss = 1 - dice_coeff
        return dice_loss.mean() # Average over depth

    def FocalTverskyLoss(self, output: torch.Tensor, target : torch.Tensor, epsilon : float = 1e-8) -> float:
        '''
        Calculate the Focal Tversky loss of a predicted mask with respect to the ground truth.

        Args:
            output (torch.Tensor): Predicted mask logits Float32 tensor (D * 2 * H * W).
            target (torch.Tensor): Ground truth binary (0,1) mask Int64 tensor (D * H * W).
            epsilon (float): The smoothing factor.
        
        Returns:
            focal_tversky_loss (float): The Focal Tversky loss.
        '''
        # Hyperparameters
        alpha = 1 - self.beta
        beta = self.beta
        gamma = self.gamma
        # Softmax the logits to probabilities
        probs = torch.softmax(output, dim = 1)
        # Excite the target tensor to shape (D * 2 * H * W)
        target_onehot = torch.nn.functional.one_hot(target, num_classes = 2).permute(0, 3, 1, 2).float()
        # Focus on foreground class (class 1)
        probs_fg = probs[:, 1, :, :]
        target_fg = target_onehot[:, 1, :, :]
        # Calculate loss coefficient
        TP = (probs_fg * target_fg).sum(dim = (1, 2)) # True Positive
        FP = (probs_fg * (1 - target_fg)).sum(dim = (1, 2)) # False Positive
        FN = ((1 - probs_fg) * target_fg).sum(dim = (1, 2)) # False Negative
        tversky_coeff = (TP + epsilon) / (TP + alpha * FP + beta * FN + epsilon)
        focal_tversky_loss = (1 - tversky_coeff)**gamma
        return focal_tversky_loss.mean() # Average over depth

def grayscale_to_rgb(scan : np.ndarray[float], cmap : str = 'inferno') -> np.ndarray[float]:
    '''
    Colours a greyscale intensity plot.

    Args:
        scan (np.ndarray): Input greyscale scan array (D * H * W).
        cmap (string): Choice of colormap, out of the matplotlib strings - grey, bone, viridis, plasma, inferno etc...

    Returns:
        scan (np.ndarray): Output RGB scan array (D * 3 * H * W).
    '''
    # Convert greyscale to cmap and multiply by 255
    cmap = plt.get_cmap(cmap)
    scan = cmap(scan/scan.max())*255 # (D * H * W * 4)
    # Convert RGBA to RGB
    scan = np.delete(scan, 3, axis = 3) # (D * H * W * 3)
    return scan

def clip_and_scale(tensor : torch.Tensor, low_clip : float = 1., high_clip : float = 99.) -> torch.Tensor:
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
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
    return tensor

def sum_IoU(pred_mask : torch.Tensor, true_mask : torch.Tensor) -> float:
    '''
    Calculate the Intersection over Union (IoU) value between the predicted mask and ground truth.

    Args:
        pred_mask (torch.Tensor): A binary (0,1) predicted mask Int64 tensor (any shape).
        true_mask (torch.Tensor): A binary (0,1) ground truth mask Int64 tensor (same shape as pred_mask)
    
    Returns:
        IoU (float): The IoU score of the two mask tensors.
    '''
    intersection = torch.logical_and(pred_mask, true_mask).sum().item()
    union = torch.logical_or(pred_mask, true_mask).sum().item()

    if union == 0: # Handle empty masks
        return 1.0 if intersection == 0 else 0.0
    return float(intersection)/union

def get_model_instance_segmentation(num_classes : int, trained : bool = False) -> torchvision.models.segmentation.fcn.FCN:
    '''
    Load an instance of the FCN ResNet 101 model, with pre-trained weights.

    Args:
        num_classes (int): The number of output classes. Here, we use two -> 0. Background | 1. Blood vessel
        trained (bool): A boolean depicting whether the model has been locally trained or not, i.e. whether to load fine-tuned or default weights.

    Returns:
        model (torchvision.models.segmentation.fcn_resnet101): The model with required weights.
    '''
    # Load the instance segmentation model pre=trained on COCO
    model = torchvision.models.segmentation.fcn_resnet101(weights = "COCO_WITH_VOC_LABELS_V1")

    # Replace the classifier with a new one, that has a user defined num_classes
    # Get the number of input features for the final layer of the ResNet
    inter_channels = model.classifier[4].in_channels
    # Replace the final convolutional layer with a new one
    model.classifier[4] = nn.Conv2d(inter_channels, num_classes, kernel_size = 1)

    if trained: # If the model has been locally trained, load the fine-tuned weights
        model.load_state_dict(torch.load('model_params.pth', weights_only = True))
    return model

def get_transform(phase : str = 'val'):
    # Need to complete. ----
    return