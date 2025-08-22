import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch
from torchvision.transforms import v2 as T
from utils.custom_transforms import ToTensor, Resize, ClipAndScale
from utils.segmentation_util import get_model_instance_unet, get_transform

val_transform = T.Compose([
    ToTensor(),
    T.ToDtype(torch.float32, scale=True),
    Resize(size=(64, 64), interpolation='bilinear'),
    ClipAndScale()
])

def evaluation(model, scan, device):
    '''
    Calculates the mask.
    Args
        model: the model being used
        images: a numpy array of pixel values (single channel, i.e. greyscale)
        device: the device being used
    Returns
        masks: a tensor of masks, the same shape as the images array
    '''
    if model.training:
        model.eval()
    with torch.inference_mode():
        inputs = val_transform(scan).to(device)  # (D * 1 * H * W)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim = 1)
    masks = preds.squeeze().cpu()
    return masks

# Device and model setup
dims = 3
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
model = get_model_instance_unet(num_classes = 2, device = device, dims = dims, trained = True)

# Load data: Optionally apply Gaussian smoothing
target = 'Carotid'
images = np.load(f'data/val/magn/{target}.npy')

# Inference
masks = evaluation(model, images, device)

# Visualization
pcm = []
fig, ax = plt.subplots(1, 2, figsize = (10,6))
pcm.append(ax[0].imshow(images[13], cmap = 'bone'))
pcm.append(ax[1].imshow(masks[13], cmap = 'cividis', vmin=0, vmax=1))
fig.colorbar(pcm[1], ax = ax, shrink = 0.6)
ax[0].axis("off")
ax[1].axis("off")

# Animation update function
def updateAnim(frame):
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