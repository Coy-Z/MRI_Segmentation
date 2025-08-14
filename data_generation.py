from utils import data_gen_util as dg
import numpy as np

def data_gen(V: dg.Random_Speed_Field, SDF: dg.SDF_MRI_Circle | dg.SDF_MRI_Tube, depth: int = 10):
    mask3D = []
    magn3D = []
    SDF.update_speed_field(V)
    SDF.step_sdf_analytical_grad(iterations = 100)
    for _ in range(depth):
        edit_flag = np.random.randint(0, 2)
        if edit_flag:
            V.reset()
            V.sinusoidal(freq_range=(0.01, 0.05), amp_range=(0, 3), num_modes=4) # High frequency, low amplitude
            V.sinusoidal(freq_range=(0.001, 0.01), amp_range=(3, 7), num_modes=2) # Low frequency, high amplitude
            V.affine(grad_range=(-0.1, 0.1), bias_range=(-1, 2))
            SDF.update_speed_field(V)
        SDF.step_sdf_analytical_grad(iterations = 20)
        mask, magn = SDF.return_mask_magn_pair()
        mask3D.append(mask)
        magn3D.append(magn)
    mask = np.array(mask3D)
    magn = np.array(magn3D)
    return mask, magn

def data_generator(num: int = 10, depth: int = 10):
    masks = []
    magnitudes = []
    for _ in range(num):
        V = dg.Random_Speed_Field((100, 100))
        V.sinusoidal(freq_range=(0.01, 0.05), amp_range=(0, 2), num_modes=4) # High frequency, low amplitude
        V.sinusoidal(freq_range=(0.001, 0.01), amp_range=(2, 5), num_modes=2) # Low frequency, high amplitude
        V.affine(grad_range=(-0.1, 0.1), bias_range=(-1, 2))
        type_rand = np.random.randint(0, 2)
        if type_rand:
            sdf = dg.SDF_MRI_Tube(V)
        else:
            sdf = dg.SDF_MRI_Circle(V)
        mask, magn = data_gen(V, sdf, depth)
        masks.append(mask)
        magnitudes.append(magn)
    return masks, magnitudes

def data_saver(masks, magnitudes, dir):
    assert len(masks) == len(magnitudes), "Masks and magnitudes must have the same length."
    for i in range(len(masks)):
        mask_path = f"{dir}/mask/artificial_mask_{i}.npy"
        magn_path = f"{dir}/magn/artificial_magn_{i}.npy"
        np.save(mask_path, masks[i])
        np.save(magn_path, magnitudes[i])

masks, magnitudes = data_generator(10, 10)
data_saver(masks, magnitudes, "C:/Users/ZHUCK/Uni/UROP25/FCNResNet_Segmentation/data/train")