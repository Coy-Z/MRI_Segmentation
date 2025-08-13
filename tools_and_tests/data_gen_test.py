import time
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_gen_util import Random_Speed_Field, SDF_MRI_Circle, SDF_MRI_Tube

since = time.time()
V = Random_Speed_Field(shape=(100, 100))
V.sinusoidal(freq_range=(0.001, 0.05), amp_range=(0, 1), num_modes=2)
V.affine(grad_range=(-0.05, 0.05), bias_range=(-1, 2))
sdf_mri_circ = SDF_MRI_Circle(V.get_speed_field(), r = 22)
sdf_mri_tube = SDF_MRI_Tube(V.get_speed_field(), r=10.)

#smooth_noise = sdf_mri_circ.gaussian_process(grid_shape=(100, 100), length_scale=0.1, variance=10.)
#initial_sdf = sdf_mri_circ.get_sdf()
#initial_sdf = np.where(sdf_mri_circ.get_sdf() < 0, 1, 0)

sdf_mri_circ.step_sdf(iterations=100)
sdf_mri_tube.step_sdf(iterations=100)
maskc, magnc = sdf_mri_circ.return_mask_magn_pair()
maskt, magnt = sdf_mri_tube.return_mask_magn_pair()

#final_sdf = sdf_mri_circ.get_sdf()

print("Time taken:", time.time() - since)

# Plot Results
fig, ax = plt.subplots(2, 2, figsize=(8, 6))
pcm = []
pcm.append(ax[0, 0].pcolormesh(maskc, cmap='viridis'))
ax[0, 0].set_title('Mask')
ax[0, 0].axis('off')
pcm.append(ax[0, 1].pcolormesh(magnc, cmap='viridis'))
ax[0, 1].set_title('Magnitude')
ax[0, 1].axis('off')
pcm.append(ax[1, 0].pcolormesh(maskt, cmap='viridis'))
ax[1, 0].set_title('Mask')
ax[1, 0].axis('off')
pcm.append(ax[1, 1].pcolormesh(magnt, cmap='viridis'))
ax[1, 1].set_title('Magnitude')
ax[1, 1].axis('off')
fig.colorbar(pcm[0], ax = ax[0, 0], shrink = 0.6)
fig.colorbar(pcm[1], ax = ax[0, 1], shrink = 0.6)
fig.colorbar(pcm[2], ax = ax[1, 0], shrink = 0.6)
fig.colorbar(pcm[3], ax = ax[1, 1], shrink = 0.6)

plt.tight_layout()
plt.show()