import time
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_gen_util import Random_Speed_Field, SDF_MRI_Circle, SDF_MRI_Tube

since = time.time()
Vc = Random_Speed_Field(shape=(100, 100))
Vt = Random_Speed_Field(shape=(100, 100))
Vc.sinusoidal(freq_range=(0.01, 0.03), amp_range=(15, 30), num_modes=2)
Vc.random_coherent(log_length_scale_mean=-2.3, log_length_scale_variance=0.5, variance=40)
Vc.random_coherent(log_length_scale_mean=0, log_length_scale_variance=1, variance=30)
Vt.random_coherent(log_length_scale_mean=-2, log_length_scale_variance=0.5, variance=10)
Vt.random_coherent(log_length_scale_mean=0, log_length_scale_variance=1, variance=40)

sdf_mri_circ = SDF_MRI_Circle(Vc, r = 22.)
sdf_mri_tube = SDF_MRI_Tube(Vt, r = 10.)

#smooth_noise = sdf_mri_circ.gaussian_process(grid_shape=(100, 100), length_scale=0.1, variance=10.)
#initial_sdf = sdf_mri_circ.get_sdf()
#initial_sdf = np.where(sdf_mri_circ.get_sdf() < 0, 1, 0)
#dn, ds, de, dw = sdf_mri_circ.get_derivatives()
#initial_nabla_pos, initial_nabla_neg = sdf_mri_circ.get_nablas()

sdf_mri_circ.step_sdf_numerical_grad(iterations=100)
sdf_mri_tube.step_sdf_numerical_grad(iterations=100)
maskc, magnc = sdf_mri_circ.return_mask_magn_pair()
maskt, magnt = sdf_mri_tube.return_mask_magn_pair()

final_sdfc = sdf_mri_circ.get_sdf()
final_sdft = sdf_mri_tube.get_sdf()

#dn, ds, de, dw = sdf_mri_circ.get_derivatives()
#final_nabla_pos, final_nabla_neg = sdf_mri_circ.get_nablas()

print("Time taken:", time.time() - since)

# Plot Results
fig, ax = plt.subplots(2, 3, figsize=(10, 6))
pcm = []
pcm.append(ax[0, 0].pcolormesh(maskc, cmap='viridis'))
ax[0, 0].set_title('Mask: Circle Seed')
ax[0, 0].axis('off')
pcm.append(ax[0, 1].pcolormesh(magnc, cmap='viridis'))
ax[0, 1].set_title('Magnitude: Circle Seed')
ax[0, 1].axis('off')
ax[0, 1].contour(final_sdfc, colors='white', linewidths=1)
pcm.append(ax[1, 0].pcolormesh(maskt, cmap='viridis'))
ax[1, 0].set_title('Mask: Tube Seed')
ax[1, 0].axis('off')
pcm.append(ax[1, 1].pcolormesh(magnt, cmap='viridis'))
ax[1, 1].set_title('Magnitude: Tube Seed')
ax[1, 1].axis('off')
ax[1, 1].contour(final_sdft, colors='white', linewidths=1)
pcm.append(ax[0, 2].pcolormesh(Vc.field, cmap='viridis'))
ax[0, 2].set_title('Speed Field: Circle Seed')
ax[0, 2].axis('off')
pcm.append(ax[1, 2].pcolormesh(Vt.field, cmap='viridis'))
ax[1, 2].set_title('Speed Field: Tube Seed')
ax[1, 2].axis('off')
fig.colorbar(pcm[0], ax = ax[0, 0], shrink = 0.6)
fig.colorbar(pcm[1], ax = ax[0, 1], shrink = 0.6)
fig.colorbar(pcm[2], ax = ax[1, 0], shrink = 0.6)
fig.colorbar(pcm[3], ax = ax[1, 1], shrink = 0.6)
fig.colorbar(pcm[4], ax = ax[0, 2], shrink = 0.6)
fig.colorbar(pcm[5], ax = ax[1, 2], shrink = 0.6)

plt.tight_layout()
plt.show()