import numpy as np
import itertools
import vtk
from paraview.simple import *
from vtk.util.numpy_support import vtk_to_numpy

paramChoices = [
    [0.05, 0.2], # alpha
    [0.05, 0.2], # beta
    [0.05, 0.2], # gamma
    [1, 3],     # omega1
    [1, 3],     # omega2
    [1, 3],     # omega3
    [1, 3],     # omega4
    [1, 3],     # omega5
    [1, 3],     # omega6
    [1, 3],     # omega7
    [1, 3],     # omega8
    [1, 3],     # omega9
]

# Iterate over a range of abg combos.
for i, (a, b, g, w1, w2, w3, w4, w5, w6, w7, w8, w9) in enumerate(itertools.product(*paramChoices)):
    # Define parameters
    alpha, beta, gamma, omega1, omega2, omega3, omega4, omega5, omega6, omega7, omega8, omega9 = a, b, g, w1, w2, w3, w4, w5, w6, w7, w8, w9  # Warp parameters
    file_name = f"Aorta_Warp_{i}"          # Output base filename
    field_names = ["magn", "in_"]       # Scalar fields to export
    output_dir = rf"C:\Users\ZHUCK\Uni\UROP25\FCNResNet_Segmentation\data\train"

    # Get the input (which is a vtkStructuredGrid)
    grid = GetActiveSource()
    grid.UpdatePipeline()
    vtk_grid = grid.GetClientSideObject().GetOutput()

    # Extract grid information
    points = vtk_grid.GetPoints()
    n_points = points.GetNumberOfPoints()
    dims = [0, 0, 0]
    vtk_grid.GetDimensions(dims)
    dx = points.GetPoint(1)[0] - points.GetPoint(0)[0]
    dy = points.GetPoint(dims[0])[1] - points.GetPoint(0)[1]
    dz = points.GetPoint(dims[0]*dims[1])[2] - points.GetPoint(0)[2]
    spacing = [dx, dy, dz]
    origin = [0, 0, 0]

    # Warp the coordinates of points
    new_points = vtk.vtkPoints()
    new_points.SetNumberOfPoints(n_points)

    for i in range(n_points):
        x, y, z = points.GetPoint(i)
        wx = x + alpha * np.sin(omega1 * y) + beta * np.cos(omega2 * z) + gamma * np.sin(omega3 * x)
        wy = y + beta  * np.sin(omega4 * z) + gamma * np.cos(omega5 * x) + alpha * np.sin(omega6 * y)
        wz = z + gamma * np.sin(omega7 * x) + alpha * np.cos(omega8 * y) + beta * np.sin(omega9 * z)

        wx *= 1.2
        wy *= 1.2
        wz *= 1.2

        new_points.SetPoint(i, wx, wy, wz)

    # Generate the warped grid
    warped_data = vtk.vtkStructuredGrid()
    warped_data.ShallowCopy(vtk_grid)
    warped_data.SetPoints(new_points)

    # Create a resampling grid, which is uniform, allowing for numpy to infer the correct coordinates
    uniform_grid = vtk.vtkImageData()
    uniform_grid.SetDimensions(dims)
    uniform_grid.SetSpacing(spacing)
    uniform_grid.SetOrigin(origin)

    # Resample Using vtkResampleWithDataSet
    resampler = vtk.vtkResampleWithDataSet()
    resampler.SetInputData(uniform_grid)
    resampler.SetSourceData(warped_data)
    resampler.Update()
    resampled_output = resampler.GetOutput()

    # Extract and save the scalar fields
    for field_name in field_names:
        vtk_array = resampled_output.GetPointData().GetArray(field_name)
        if not vtk_array:
            print(f"[WARN] Field '{field_name}' not found.")
            continue
        
        # Convert to NumPy
        
        flat_array = vtk_to_numpy(vtk_array)
        # Reshape to 3D (VTK is x-fastest: Fortran order)
        reshaped_array = flat_array.reshape(tuple(dims), order='F')

        loc = "magn" if field_name == "magn" else "mask"
        np.save(fr"{output_dir}\{loc}\{file_name}.npy", reshaped_array)
        np.save(fr"{output_dir}\{loc}\{file_name}.npy", reshaped_array)
        print(f"Saved '{field_name}' with shape {reshaped_array.shape} to {file_name}.npy")