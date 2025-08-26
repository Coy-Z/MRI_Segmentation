import numpy as np
from paraview.simple import *
from vtk.util.numpy_support import vtk_to_numpy

# Get the structured grid
grid = GetActiveSource()
grid.UpdatePipeline()
vtk_grid = grid.GetClientSideObject().GetOutput()

# Choose your scalar field
for field_name in ["in_"]:
    vtk_array = vtk_grid.GetPointData().GetArray(field_name)

    # Convert to NumPy
    flat_array = vtk_to_numpy(vtk_array)

    # Get dimensions of the structured grid
    dims = [0, 0, 0]
    vtk_grid.GetDimensions(dims)
    dims = tuple(dims)  # e.g. (188, 84, 197)

    # Reshape to 3D (VTK is x-fastest: Fortran order)
    reshaped_array = flat_array.reshape(dims, order='F')

    file_name = 'Aorta'
    if field_name == "magn":
        loc = "magn"
    else: loc = "mask"
    # Save to .npy
    np.save(rf"C:\Users\ZHUCK\Uni\UROP25\FCNResNet_Segmentation\data\train\{loc}\{file_name}.npy", reshaped_array)
    #np.save(rf"workspace\fcn_resnet_MRI_seg\data\train\{loc}\{file_name}.npy", reshaped_array)
    #np.save(rf"workspace\fcn_resnet_MRI_seg\data\val\{loc}\{file_name}.npy", reshaped_array)


    print(f"Saved {field_name}.npy with shape {reshaped_array.shape}")
