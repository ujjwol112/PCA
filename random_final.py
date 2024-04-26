# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)                                      # Set random seed for reproducibility of dataset
matA = np.random.randn(20, 2)                           # Generate random data matrix from normal distribution

# Plot original data
plt.scatter(matA[:, 0], matA[:, 1])
plt.title("THA076BEI040, THA076BEI042", fontsize=10)
plt.suptitle("Original Data", fontweight='bold')
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

matB = np.random.uniform(0, 1, (2, 2))                  # Generate random transformation matrix from uniform distribution
matC = np.matmul(matA, matB)                            # Transform the data

# Plot transformed data
plt.scatter(matC[:, 0], matC[:, 1])
plt.title("THA076BEI040, THA076BEI042", fontsize=10)
plt.suptitle("Transformed Data", fontweight='bold')
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

var_x = np.var(matC[:, 0])                              # Variance along data axis
var_y = np.var(matC[:, 1])                              # Variance along axis perpendicular to data axis
print("Variance along X-axis:", var_x)
print("Variance along Y-axis:", var_y)

# covariance matrix calculation using the formula & in-built function
covMat = (1 / (len(matC) - 1)) * np.dot(matC.T, matC)   # 1/(n-1) XX^T
temp = np.cov(matC.T)                                   # Covariance matrix for comparision with direct formula

# Calculate eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(temp)
pro_var = eigenvalues / np.sum(eigenvalues)             # Proportion of variance calculation

# Perform PCA transformation
matY = np.matmul(eigenvectors.T, matC.T)
covMatY = np.cov(matY)
matY = matY.T

# Plot PCA data
plt.scatter(matY[:, 0], matY[:, 1])
plt.title("THA076BEI040, THA076BEI042", fontsize=10)
plt.suptitle("PCA Data", fontweight='bold')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.axis('equal')
plt.grid(True)
plt.show()

# Perform PCA transformation to 1D
pca_1d = np.matmul(eigenvectors.T[:, 0], matC.T)

# Plot 1D PCA data
plt.scatter(pca_1d, np.zeros_like(pca_1d))
plt.title("THA076BEI040, THA076BEI042", fontsize=10)
plt.suptitle("PCA Data (1D)", fontweight='bold')
plt.xlabel("PC1")
plt.axis('equal')
plt.grid(True)
plt.show()
