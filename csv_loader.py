import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation

scan = np.loadtxt('data/data2/ns_00_vtk_out.csv', delimiter = ',', skiprows = 1)
print(scan.shape)