import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation

filename = 'Coarct_Aorta'
scanMagn = np.load(f'data/val/magn/{filename}.npy')
scanMask = np.load(f'data/val/mask/{filename}.npy')
scan = {'Magnitude' : scanMagn, 'Mask' : scanMask}
keys = ['Magnitude', 'Mask']
numSlicesMagn = scanMagn.shape[0]
numSlicesMask = scanMask.shape[0]

fig, ax = plt.subplots(2, 2, figsize = (10, 7))
plt.subplots_adjust(bottom=0.2)
title = [[None, None],[None, None]]
pcm = [[None, None],[None, None]]
for i in range(2):
    for j in range(2):
        #ax[i,j].set_aspect('equal')
        pcm[i][j] = ax[i,j].pcolormesh(scan[keys[j]][0], cmap = 'viridis')
        pcm[i][j].set_clim(vmin = scan[keys[j]].min(), vmax = scan[keys[j]].max())
        title[i][j] = ax[i,j].set_title(f'{keys[j]} MRI Slice 0')
fig.colorbar(pcm[0][0], ax = ax[0, 0], shrink = 0.6)
fig.colorbar(pcm[0][1], ax = ax[0, 1], shrink = 0.6)
fig.colorbar(pcm[1][0], ax = ax[1, 0], shrink = 0.6)
fig.colorbar(pcm[1][1], ax = ax[1, 1], shrink = 0.6)
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, label = 'Slice', valmin = 0, valmax = min(numSlicesMagn, numSlicesMask) - 1, valinit = 0, valstep = 1)
ax[0, 0].axis('off')
ax[0, 1].axis('off')
ax[1, 0].axis('off')
ax[1, 1].axis('off')

# Update function
def updateSlide(val):
    iter = int(slider.val)
    for i in range(2):
        pcm[1][i].set_array(scan[keys[i]][iter].ravel())
        #pcm[1][i].set_clim(vmin = scan[keys[i]][iter].min(), vmax = scan[keys[i]][iter].max())  # Optional: update color scale
        title[1][i].set_text(f"{keys[i]} MRI Slice {iter}")
    fig.canvas.draw_idle()

# Animation update function
def updateAnim(frame):
    for i in range(2):
        pcm[0][i].set_array(scan[keys[i]][frame].ravel())
        #pcm[0][i].set_clim(vmin = scan[keys[i]][frame].min(), vmax = scan[keys[i]][frame].max())  # Optional: update color scale
        title[0][i].set_text(f"{keys[i]} MRI Slice {frame}")
    return pcm[0], title[0]

# Create animation
slider.on_changed(updateSlide)

ani = FuncAnimation(fig, updateAnim, frames = min(numSlicesMagn, numSlicesMask), interval = 100, blit = False)

# Save animation as GIF to prevent the warning and ensure it's properly rendered
#print("Saving animation as GIF...")
#ani.save('images/thin_tube_thin_edge_resolved.gif', writer='pillow', fps=10)
#print("Animation saved as thin_tube_thin_edge_resolved.gif")

plt.show()

