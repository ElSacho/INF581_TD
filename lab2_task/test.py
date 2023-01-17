import numpy as np

xy = np.array([1,2,3,4])
xy = xy.reshape(1,-1) 

y = np.zeros(10)

xy = np.append(xy, y[:2])
xy = xy.reshape(1,-1) 
print(xy.shape)