import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors

'''
Note:
    data['STR'].item()[0] is the mask.
    data['STR'].item()[3, 4, 5, 6] are the spacial data, of which magnitude can be found by taking the absolute value.
    For the highest quality magnitude data, we should average across [3, 4, 5, 6] and across x, y and z.
'''

datax = scipy.io.loadmat(rf'C:\Users\ZHUCK\Uni\UROP25\FCNResNet_Segmentation\data\aorta_low_re\RawData\256x128x128\x')
datay = scipy.io.loadmat(rf'C:\Users\ZHUCK\Uni\UROP25\FCNResNet_Segmentation\data\aorta_low_re\RawData\256x128x128\y')
dataz = scipy.io.loadmat(rf'C:\Users\ZHUCK\Uni\UROP25\FCNResNet_Segmentation\data\aorta_low_re\RawData\256x128x128\z')

# Print keys to understand the structure of the loaded data
print(datax.keys())

# Extract the 'STR' field from each data dictionary (the numerical data)
str_datax = datax['STR']
str_datay = datay['STR']
str_dataz = dataz['STR']
str_tuplex = str_datax.item()
str_tupley = str_datay.item()
str_tuplez = str_dataz.item()

# Print shapes and data types of the loaded arrays
for i in range(len(str_tuplex)):
    print(f"Shape of str_tuplex[{i}]: {str_tuplex[i].shape}, Data type: {str_tuplex[i].dtype}")

def compare_xyz():
    # Compare the magnitude data from x, y and z data files
    magn = []
    for i in [3, 4, 5, 6]:
        magn.append(np.abs(str_tuplex[i]) - np.abs(str_tuplez[i]))
    numSlices = 256

    fig, ax = plt.subplots(2, 2, figsize = (8,8))
    pcm = []
    for i in range(4):
        pcm.append(ax[i//2, i%2].pcolormesh(magn[i][128], cmap = 'cividis'))
        ax[i//2, i%2].set_title(f'str_tuple[{i+3}]')
        ax[i//2, i%2].set_axis_off()
        fig.colorbar(pcm[i], ax=ax[i//2, i%2])

    # Animation update function
    def updateAnim(frame : int):
        for i in range(4):
            pcm[i].set_array(magn[i][frame].ravel())
        return pcm

    ani = FuncAnimation(fig, updateAnim, frames = numSlices, interval = 100, blit = False)

    plt.show()

def display_xdata_single(i: int):
    # Display the x data for a specific index
    magn = np.abs(str_tuplex[i] - str_tuplex[i + 4]) # Subtracting off background noise
    numSlices = magn.shape[0]
    fig, ax = plt.subplots(1, 1, figsize = (6, 6))
    pcm = ax.pcolormesh(magn[128], cmap = 'cividis')
    ax.set_title(f'str_tuple[{i}]')
    ax.set_axis_off()
    fig.colorbar(pcm, ax = ax, extend = 'max')

    # Animation update function
    def updateAnim(frame : int):
        pcm.set_array(magn[frame].ravel())
        return pcm
    
    ani = FuncAnimation(fig, updateAnim, frames = numSlices, interval = 50, blit = False)
    plt.show()

def display_xdata(arr: list):
    # Display the x data for multiple indices
    magn = []
    for i in arr:
        magn.append(np.abs(str_tuplex[i]))
    numSlices = 256

    fig, ax = plt.subplots(2, (len(arr) + 1)//2, figsize=(len(arr) * 4, 8))
    pcm = []
    for i in range(len(arr)):
        pcm.append(ax[i%2, i//2].pcolormesh(magn[i][128], cmap = 'cividis'))
        ax[i%2, i//2].set_title(f'str_tuple[{arr[i]}]')
        ax[i%2, i//2].set_axis_off()
        fig.colorbar(pcm[i], ax=ax[i%2, i//2])

    # Animation update function
    def updateAnim(frame : int):
        for i in range(len(arr)):
            pcm[i].set_array(magn[i][frame].ravel())
        return pcm

    ani = FuncAnimation(fig, updateAnim, frames = numSlices, interval = 50, blit = False)

    plt.show()

def compare_masks():
    # Compare the mask data from x, y and z data files
    mask = np.abs(str_tuplex[0]) - np.abs(str_tuplez[0])
    numSlices = 256

    fig, ax = plt.subplots(1, 1, figsize = (8,8))
    pcm = ax.pcolormesh(mask[128], cmap = 'cividis')
    ax.set_title(f'Mask Comparison')
    ax.set_axis_off()
    fig.colorbar(pcm, ax = ax)

    # Animation update function
    def updateAnim(frame : int):
        pcm.set_array(mask[frame].ravel())
        return pcm

    ani = FuncAnimation(fig, updateAnim, frames = numSlices, interval = 100, blit = False)

    plt.show()

#compare_masks()
#compare_xyz()
#display_xdata_single(7)
#display_xdata([3, 4, 5, 6, 7, 8, 9, 10])

sumMagn = np.zeros_like(str_tuplex[0], dtype = np.float32)
sumMask = np.zeros_like(str_tuplex[0], dtype = np.int8)
counter = 0
for str_tuple in [str_tuplex, str_tupley, str_tuplez]:
    for i in range(3, 7):
        sumMagn += np.abs(str_tuple[i])
        print(i)
        counter += 1
    sumMask += str_tuple[0]

magn = sumMagn/counter
mask = np.rint(sumMask/3.).astype(np.int8)
print(magn.dtype)
print(mask.dtype)

for loc in ['train', 'val']:
    np.save(rf"C:\Users\ZHUCK\Uni\UROP25\FCNResNet_Segmentation\data\{loc}\mask\Aorta.npy", mask)
    np.save(rf"C:\Users\ZHUCK\Uni\UROP25\FCNResNet_Segmentation\data\{loc}\magn\Aorta.npy", magn)