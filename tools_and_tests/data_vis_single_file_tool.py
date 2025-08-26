import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation

scan = np.load('data/mask.npy')
print(scan.shape)
numSlices = scan.shape[0]
fig, ax = plt.subplots(1, 2, figsize = (14, 7))
plt.subplots_adjust(bottom=0.2)

pcm = []
for i in range(2):
    pcm.append(ax[i].pcolormesh(scan[0], cmap = 'viridis'))
    pcm[i].set_clim(vmin = scan.min(), vmax = scan.max())

fig.colorbar(pcm[1], ax = ax, shrink = 0.6)
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, label = 'Slice', valmin = 0, valmax = numSlices - 1, valinit = 0, valstep = 1)

# Update function
def updateSlide(val):
    iter = int(slider.val)
    pcm[1].set_array(scan[iter].ravel())
    fig.canvas.draw_idle()

# Animation update function
def updateAnim(frame : int):
    pcm[0].set_array(scan[frame].ravel())
    return pcm[0]

# Create animation
slider.on_changed(updateSlide)

ani = FuncAnimation(fig, updateAnim, frames = numSlices, interval = 100, blit = False)

plt.show()