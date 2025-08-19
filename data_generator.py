from utils import data_gen_util as dg
import numpy as np

def data_gen(V : dg.Random_Speed_Field, SDF : dg.SDF_MRI_Circle | dg.SDF_MRI_Tube, depth : int = 100) -> tuple[np.ndarray, np.ndarray]:
    mask3D = []
    magn3D = []
    SDF.update_speed_field(V)
    SDF.step_sdf_analytical_grad(iterations = 100)
    for _ in range(depth):
        edit_flag = np.random.randint(0, 2)
        if edit_flag:
            V.reset()
            if type(SDF) == dg.SDF_MRI_Circle:
                V.sinusoidal(freq_range=(0.01, 0.03), amp_range=(15, 30), num_modes=2)
                V.random_coherent(log_length_scale_mean=-2, log_length_scale_variance=0.5, amplitude_variance=20)
                V.random_coherent(log_length_scale_mean=0, log_length_scale_variance=1, amplitude_variance=30)
                V.affine(grad_range=(-0.05, 0.05), bias_range=(0, 0))
            else: # type is dg.SDF_MRI_Tube
                V.random_coherent(log_length_scale_mean=-2, log_length_scale_variance=0.5, amplitude_variance=10)
                V.random_coherent(log_length_scale_mean=0, log_length_scale_variance=1, amplitude_variance=40)
                V.affine(grad_range=(-0.05, 0.05), bias_range=(-1, 0))
            SDF.update_speed_field(V)
        SDF.step_sdf_analytical_grad(iterations = 10)
        #SDF.step_sdf_numerical_grad(iterations = 10)
        mask, magn = SDF.return_mask_magn_pair()
        mask3D.append(mask)
        magn3D.append(magn)
    mask = np.array(mask3D)
    magn = np.array(magn3D)
    return mask, magn

def data_generator(num : int = 10, depth : int = 100) -> tuple[list[np.ndarray], list[np.ndarray]]:
    masks = []
    magnitudes = []
    for _ in range(num):
        V = dg.Random_Speed_Field((100, 100))
        V.affine(grad_range=(-0.1, 0.1), bias_range=(-1.5, 2))
        type_rand = np.random.randint(0, 2)
        if type_rand:
            V.random_coherent(log_length_scale_mean=-2, log_length_scale_variance=0.5, amplitude_variance=10)
            V.random_coherent(log_length_scale_mean=0, log_length_scale_variance=1, amplitude_variance=40)
            sdf = dg.SDF_MRI_Tube(V, smoothed = True)
        else:
            V.sinusoidal(freq_range=(0.01, 0.03), amp_range=(15, 30), num_modes=2)
            V.random_coherent(log_length_scale_mean=-2.3, log_length_scale_variance=0.5, amplitude_variance=40)
            V.random_coherent(log_length_scale_mean=0, log_length_scale_variance=1, amplitude_variance=30)
            sdf = dg.SDF_MRI_Circle(V)
        mask, magn = data_gen(V, sdf, depth)
        masks.append(mask)
        magnitudes.append(magn)
    return masks, magnitudes

def data_saver(masks : list[np.ndarray], magnitudes : list[np.ndarray], dir: str):
    assert len(masks) == len(magnitudes), "Masks and magnitudes must have the same length."
    for i in range(len(masks)):
        mask_path = f"{dir}/mask/artificial_{i}.npy"
        magn_path = f"{dir}/magn/artificial_{i}.npy"
        np.save(mask_path, masks[i])
        np.save(magn_path, magnitudes[i])

masks, magnitudes = data_generator(1000, 100)
data_saver(masks, magnitudes, "./data/train")