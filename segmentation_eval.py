import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch
from torchvision.transforms import v2 as T
from utils.custom_transforms import ToTensor, Resize, ClipAndScale
from utils.segmentation_util import get_model_instance_unet, get_transform

def evaluation(model, dims : int, scan, transform, device):
    '''
    Calculates the mask.
    Args:
        model: The model being used.
        dims (int): The number of dimensions.
        scan: A numpy array of pixel values (single channel, i.e. greyscale).
        transform: The transform to apply.
        device: The device being used.

    Returns:
        torch.Tensor: A tensor of masks, the same shape as the images array.
    '''
    if model.training:
        model.eval()
    with torch.inference_mode():
        inputs = transform(scan).to(device)  # (D * H * W)
        inputs = inputs.unsqueeze(3 - dims)  # 2D (D * 1 * H * W) or 3D (1 * D * H * W)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim = 3 - dims)
    masks = preds.squeeze().cpu()
    return masks

# Device and model setup
dims = 3
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
model = get_model_instance_unet(num_classes = 2, device = device, dims = dims, trained = True)

# Define validation transform
val_transform = T.Compose([
    ToTensor(),
    T.ToDtype(torch.float32, scale=True),
    Resize(dims = dims, size = (64, 64) if dims == 2 else (64, 64, 64), interpolation = 'bilinear' if dims == 2 else 'trilinear'),
    ClipAndScale()
])

# Load data
target = 'Coarct_Aorta'
images = np.load(f'data/val/magn/{target}.npy')

# Inference
masks = evaluation(model, dims, images, val_transform, device)

# Resize images
images = Resize(dims = dims, size = (64, 64) if dims == 2 else (64, 64, 64), interpolation = 'bilinear' if dims == 2 else 'trilinear')(ToTensor()(images)).numpy()

# Visualization
pcm = []
fig, ax = plt.subplots(1, 2, figsize = (10,6))
pcm.append(ax[0].imshow(images[13], cmap = 'bone'))
pcm.append(ax[1].imshow(masks[13], cmap = 'cividis', vmin=0, vmax=1))
fig.colorbar(pcm[1], ax = ax, shrink = 0.6)
ax[0].axis("off")
ax[1].axis("off")

# Animation update function
def updateAnim(frame : int):
    pcm[0].set_data(images[frame])
    pcm[1].set_data(masks[frame])
    return pcm

# Create animation and keep reference to prevent garbage collection
ani = FuncAnimation(fig, updateAnim, frames = images.shape[0], interval = 100, blit = False)

# Save animation as GIF to prevent the warning and ensure it's properly rendered
print("Saving animation as GIF...")
ani.save(f'images/{dims}D_{target}.gif', writer='pillow', fps=10)
print(f"Animation saved as {dims}D_{target}.gif")

# Optionally show the plot if display is available
try:
    plt.show(block=True)
except:
    print("Display not available, animation saved to file instead")