import torch
import numpy as np
import torchvision.transforms.v2 as T
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.custom_transforms import ToTensor, Resize, GaussianBlur, ClipAndScale, RandomXFlip, RandomYFlip, RandomZFlip, RandomRotation, GaussianNoise

delta = np.zeros((20, 20, 20)) # (D * H * W)
delta[10, 3, 3] = 100

dims = 2

transform = T.Compose([
    # Transform
    ToTensor(), # Good
    T.ToDtype(torch.float32, scale=True),
    Resize(dims = dims, size=(64, 64) if dims == 2 else (20, 64, 64), interpolation='nearest'), # Good
    GaussianBlur(dims = dims, kernel_size= 10, sigma = 3), # Good
    Resize(dims = dims, size=(128, 128) if dims == 2 else (20, 128, 128), interpolation = 'bilinear' if dims == 2 else 'trilinear'), # Good
    ClipAndScale(dims = dims, low_clip=1, high_clip=99, epsilon=1e-8), # Good

    # Augment
    RandomXFlip(p=1), # Good
    RandomYFlip(p=1), # Good
    #RandomZFlip(p=0.5) if dims == 3 else None, # Good
    RandomRotation(degrees=15), # Good
    GaussianNoise(mean=0, sigma=0.1, clip=True), # Good
])

tensor = transform(delta)

print(tensor.shape)

fig, ax = plt.subplots(1, 1, figsize = (10,6))
pcm = ax.pcolormesh(tensor[10], cmap = 'viridis')
fig.colorbar(pcm, ax = ax, shrink = 0.6)
title = ax.set_title(f'Frame 5')

# Animation update function
def updateAnim(frame):
    pcm.set_array(tensor[frame].ravel())
    title.set_text(f'Frame {frame}')
    return pcm, title

ani = FuncAnimation(fig, updateAnim, frames = tensor.shape[0], interval = 100, blit = False)

plt.show()

