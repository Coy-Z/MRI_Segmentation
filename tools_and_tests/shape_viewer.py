import numpy as np

filename = 'Aorta.npy'
array1 = np.load(f'data/val/mask/{filename}')
array2 = np.load(f'data/val/magn/{filename}')   
print(f"Shape of the loaded arrays: {array1.shape}, {array2.shape}")