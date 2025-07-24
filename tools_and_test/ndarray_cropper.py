import numpy as np

arr = np.load('../data/data2/mag.npy')[67:67+134,140:140+155,:65]

np.save('../data/magnAorta.npy', arr, allow_pickle = False)