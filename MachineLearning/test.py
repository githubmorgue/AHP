import torch

# 创建一个随机方阵
size = 4
matrix = torch.randn(size, size)

# 计算特征值
eigenvalues = torch.linalg.eigvals(matrix)

print("Eigenvalues:")
print(eigenvalues)

# 计算特征值和特征向量
eigenvalues_complex, eigenvectors_complex = torch.linalg.eig(matrix)

print("\nEigenvalues (Complex):")
print(eigenvalues_complex)

print("\nEigenvectors (Complex):")
print(eigenvectors_complex)

# 选择一个特征值的索引
target_eigenvalue_index = 0

# 选择对应的特征向量
target_eigenvector = eigenvectors_complex[:, target_eigenvalue_index]

# 将特征向量转换为实数类型
target_eigenvector = target_eigenvector.real

# 要检查的向量
vector = torch.randn(size)

# 检查向量是否是特征向量
is_eigenvector = torch.allclose(torch.matmul(matrix, target_eigenvector), eigenvalues_complex[target_eigenvalue_index].real * target_eigenvector)

print("\nIs the vector an eigenvector?")
print(is_eigenvector)