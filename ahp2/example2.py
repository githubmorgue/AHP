import numpy as np
from ahp2.ahp import AHP


# 准则重要性矩阵
criteria = np.array([
        [1, 3, 2], 
        [1/3, 1, 1/2], 
        [1/2, 2, 1]])

# 对每个准则，方案优劣排序
b1 = np.array([
        [1, 3, 2], 
        [1/3, 1, 1/2], 
        [1/2, 2, 1]])

b2 = np.array([
        [1, 1/2, 1/3], 
        [2, 1, 1/2], 
        [3, 2, 1]])

b3 = np.array([
        [1, 1/3, 1/2], 
        [3, 1, 2], 
        [2, 1/2, 1]])



b = [b1, b2, b3]
a = AHP(criteria, b).run()