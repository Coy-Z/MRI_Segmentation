import numpy as np
import vtk

input_data = self.GetInput()
output = self.GetOutput()

print(input_data.GetClassName())

# Extract input grid info
points = input_data.GetPoints()
n_points = points.GetNumberOfPoints()
dims = [0, 0, 0]
input_data.GetDimensions(dims)
dx = points.GetPoint(1)[0] - points.GetPoint(0)[0]
dy = points.GetPoint(dims[0])[1] - points.GetPoint(0)[1]
dz = points.GetPoint(dims[0]*dims[1])[2] - points.GetPoint(0)[2]
spacing = [dx, dy, dz]
origin = [0, 0, 0]

# Create new warped points
new_points = vtk.vtkPoints()
new_points.SetNumberOfPoints(n_points)

alpha, beta, gamma = 0.3, 0.3, 0.3

for i in range(n_points):
    x, y, z = points.GetPoint(i)
    wx = x + alpha * np.sin(0.8 * np.pi * y)
    wy = y + beta  * np.sin(0.8 * np.pi * z)
    wz = z + gamma * np.sin(0.8 * np.pi * x)

    wx *= 1.2
    wy *= 1.2
    wz *= 1.2

    new_points.SetPoint(i, wx, wy, wz)

# Create warped data as a shallow copy
warped_data = vtk.vtkStructuredGrid()
warped_data.ShallowCopy(input_data)
warped_data.SetPoints(new_points)

# Create vtkImageData for resampling grid with same geometry as input
uniform_grid = vtk.vtkImageData()
uniform_grid.SetDimensions(dims)
uniform_grid.SetSpacing(spacing)
uniform_grid.SetOrigin(origin)

# Use vtkResampleWithDataset
resampler = vtk.vtkResampleWithDataSet()
resampler.SetInputData(uniform_grid)
resampler.SetSourceData(warped_data)
resampler.Update()

# Output result to pipeline
output.ShallowCopy(resampler.GetOutput())