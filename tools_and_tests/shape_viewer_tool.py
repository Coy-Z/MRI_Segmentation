import numpy as np

filename = 'Coarct_Aorta.npy'
array1 = np.load(f'data/train/mask/{filename}')
array2 = np.load(f'data/train/magn/{filename}')   
print(f"Shape of the loaded arrays: {array1.shape}, {array2.shape}")