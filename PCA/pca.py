import numpy as np

class PCA():
    def pca(self, X, k):
        # Calculate the mean of each feature
        mean = np.mean(X, axis=0)

        # Normalize the dataset
        norm_X = X - mean

        # Compute the covariance matrix
        cov_matrix = np.cov(norm_X, rowvar=False)

        # Calculate the eigenvalues and eigenvectors
        eig_val, eig_vec = np.linalg.eigh(cov_matrix)

        # Sort eigenvectors based on eigenvalues in descending order
        indices = np.argsort(eig_val)[::-1]
        sorted_eig_vec = eig_vec[:, indices]

        # Select the top k eigenvectors
        feature = sorted_eig_vec[:, :k]

        # Transform the data
        data = np.dot(norm_X, feature)

        return data


def main():
    X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    print(PCA().pca(X, 1))

if __name__ == "__main__":
    main()