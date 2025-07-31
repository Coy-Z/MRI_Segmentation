import numpy as np
import vtk
from paraview.simple import *
from vtk.util.numpy_support import vtk_to_numpy

for i in range(8):
    # Generate random parameters for warping
    generator = np.random.default_rng(seed=i)
    alpha1, alpha2, alpha3, beta1, beta2, beta3, gamma1, gamma2, gamma3 = generator.multivariate_normal(
        mean = 0.1 * np.ones(9),
        cov = 0.05 * np.eye(9)
    )
    omega1, omega2, omega3, omega4, omega5, omega6, omega7, omega8, omega9 = np.abs(generator.multivariate_normal(
        mean = 2.0 * np.ones(9),
        cov = 0.5 * np.eye(9)
    ))
    file_name = f"Coarct_Aorta_Warp_{i}"          # Output base filename
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
        wx = x + alpha1 * np.sin(omega1 * y) + alpha2 * np.cos(omega4 * z) + alpha3 * np.sin(omega7 * x)
        wy = y + beta1 * np.sin(omega2 * z) + beta2 * np.cos(omega5 * x) + beta3 * np.sin(omega8 * y)
        wz = z + gamma1 * np.sin(omega3 * x) + gamma2 * np.cos(omega6 * y) + gamma3 * np.sin(omega9 * z)

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