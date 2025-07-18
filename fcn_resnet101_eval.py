import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2 as T
from fcn_resnet101_util import clip_and_scale, get_model_instance_segmentation, grayscale_to_rgb

val_transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Lambda(clip_and_scale)
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
        slices = grayscale_to_rgb(scan)
        inputs = torch.stack([val_transform(slice) for slice in slices]).to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs['out'], dim = 1)
    masks = preds.squeeze().cpu()
    return masks

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

model = get_model_instance_segmentation(num_classes = 2, trained = True)
model.to(device)

images = np.load('data/carotid_mask_and_magn_coy/magn.npy')

masks = evaluation(model, images, device)

# ---- Graphic display of results ----
pcm = []
fig, ax = plt.subplots(1, 2, figsize = (10,6))
pcm.append(ax[0].imshow(images[13], cmap = 'bone'))
pcm.append(ax[1].imshow(masks[13], cmap = 'cividis'))
fig.colorbar(pcm[1], ax = ax, shrink = 0.6)
ax[0].axis("off")
ax[1].axis("off")

# Animation update function
def updateAnim(frame):
    pcm[0].set_data(images[frame])
    pcm[1].set_data(masks[frame])
    return pcm

ani = FuncAnimation(fig, updateAnim, frames = images.shape[0], interval = 100, blit = False)
plt.show()