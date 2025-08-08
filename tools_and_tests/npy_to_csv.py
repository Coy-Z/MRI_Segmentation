import numpy as np

magn = np.load('data/train/magn/Aorta.npy')
mask = np.load('data/train/mask/Aorta.npy')
numPoints = magn.shape[0] * magn.shape[1] * magn.shape[2]
bounds = [(0, magn.shape[0]), (0, magn.shape[1]), (0, magn.shape[2])]

assert magn.shape == mask.shape, "Magnitude and mask shapes do not match."

# Create a combined table for magn and mask
# Each point will have 5 values: x, y, z, the magnitude, and the mask value
table = np.zeros((numPoints, 6), dtype=np.float32)

# Fill the table with coordinates and values
point_idx = 0
for x in range(magn.shape[0]):
    for y in range(magn.shape[1]):
        for z in range(magn.shape[2]):
            table[point_idx, 0] = point_idx  # Point index
            table[point_idx, 1] = x * 0.02 # x coordinate, * 0.02 to scale down
            table[point_idx, 2] = y * 0.02 # y coordinate  
            table[point_idx, 3] = z * 0.02 # z coordinate
            table[point_idx, 4] = magn[x, y, z]  # magnitude value
            table[point_idx, 5] = mask[x, y, z]  # mask value
            point_idx += 1

# Save the table to a CSV file
output_file = 'data/aorta_low_re/RawData/256x128x128/magn_mask_table.csv'
np.savetxt(output_file, table, delimiter=',', header='Point_ID,x,y,z,magn,in_', comments='', fmt='%.6f')