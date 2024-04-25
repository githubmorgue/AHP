import numpy as np
from pca import PCA 

x = np.array([
        [1, 1/5, 1/7, 2, 5], 
        [5, 1, 1 / 2, 6, 8], 
        [7, 2, 1, 7, 9],
        [1/2, 1/6, 1/7, 1, 4],
        [1/5, 1/8, 1/9, 1/4, 1]])

print(PCA().pca(x, 1))