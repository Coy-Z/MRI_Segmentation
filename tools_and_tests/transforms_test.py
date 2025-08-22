import numpy as np
import torch
import torchvision.transforms.v2 as T
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.custom_transforms import ToTensor, Resize, GaussianBlur, ClipAndScale

array = np.ones((10, 20, 30)) # (D * H * W)
delta_arr = np.zeros((10, 10, 10)) # Delta array for testing GaussianBlur
delta_arr[5, 5, 5] = 10  # Add a delta at the center

transform = T.Compose([
    ToTensor(),
    T.ToDtype(torch.float32, scale=True),
    Resize(size=(62, 64, 64), dims=3, interpolation='trilinear'),
    GaussianBlur(kernel_size = 5, sigma = 0.1),
    #ClipAndScale()
])

tensor = transform(array)

print(tensor.shape)