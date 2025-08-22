import torch
import numpy as np
import torchvision.transforms.v2 as T
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.custom_transforms import ToTensor, Resize, GaussianBlur, ClipAndScale

delta = np.zeros((20, 20, 20)) # (D * H * W)
delta[10, 10, 10] = 100

transform = T.Compose([
    ToTensor(),
    T.ToDtype(torch.float32, scale=True),
    Resize(dims = 3, size=(20, 64, 64), interpolation='nearest'),
    GaussianBlur(dims = 3, kernel_size= 10, sigma = 3),
    Resize(dims = 3, size=(20, 128, 128), interpolation = 'trilinear'),
    ClipAndScale(dims = 3, low_clip=1, high_clip=99, epsilon=1e-8)
])

tensor = transform(delta)

print(tensor.shape)

fig, ax = plt.subplots(1, 1, figsize = (10,6))
pcm = ax.pcolormesh(tensor[10], cmap = 'viridis')
fig.colorbar(pcm, ax = ax, shrink = 0.6)
title = ax.set_title(f'Frame 5')
ax.axis("off")

# Animation update function
def updateAnim(frame):
    pcm.set_array(tensor[frame].ravel())
    title.set_text(f'Frame {frame}')
    return pcm, title

ani = FuncAnimation(fig, updateAnim, frames = tensor.shape[0], interval = 100, blit = False)

plt.show()

