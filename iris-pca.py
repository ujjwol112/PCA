#import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()                          # Load the iris dataset
X = iris.data                               # Assign features to X
y = iris.target                             # Assign Targets to y
target_names = iris.target_names            # Get names of the target classes
targets = np.unique(y)                      # Extract unique target labels

# Plot the original data
plt.figure()

#Plot data individually for three different target classes
#Selects values from first column corresponding to a class on X-axis, values from second column corresponding to same class for Y-axis
plt.scatter(X[y == targets[0], 0], X[y == targets[0], 1], marker = '*',label=target_names[targets[0]])      
plt.scatter(X[y == targets[1], 0], X[y == targets[1], 1], marker = '+',label=target_names[targets[1]])      #marker = '' sets different marker style for different class
plt.scatter(X[y == targets[2], 0], X[y == targets[2], 1], marker = '.',label=target_names[targets[2]])      #label = ... Adds target names as label

plt.xlabel('Sepal Length (cm)')                                     #Adds labels
plt.ylabel('Sepal Width (cm)')
plt.suptitle("Original Iris Data", fontweight='bold')                
plt.axis('equal')                                                   # Sets aspect ratio of plot to be equal
plt.legend()                                                        # Adds legend to plot

# Perform PCA using formulas
X_stand = (X - np.mean(X, axis=0)) / np.std(X, axis=0)              # Step 1: Standardize the data
cov_matrix = (1/(len(X_stand)-1))*np.dot(X_stand.T, X_stand)        # Step 2: Compute the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)               # Step 3: Compute the eigenvalues and eigenvectors
ind = np.argsort(eigenvalues)[::-1]                                 # Step 4: Sort eigenvalues and eigenvectors
eigVal_sorted = eigenvalues[ind]                                        # Sorts eigenvalues in terms of indices
eigVec_sorted = eigenvectors[:, ind]

#Project data into new 2D space with various eigenvectors combinations
for i in range(3):
    X_pca = np.dot(X_stand, eigVec_sorted[:, i:(i+2)])        # Step 5: Project the data onto the new feature space

    # Plot the PCA data
    plt.figure()
    plt.scatter(X_pca[y == targets[0], 0], X_pca[y == targets[0], 1], marker = '*', label=target_names[targets[0]])
    plt.scatter(X_pca[y == targets[1], 0], X_pca[y == targets[1], 1], marker = '+', label=target_names[targets[1]])
    plt.scatter(X_pca[y == targets[2], 0], X_pca[y == targets[2], 1], marker = '.', label=target_names[targets[2]])
    plt.xlabel('PC'+str(i+1))
    plt.ylabel('PC'+str(i+2))
    plt.suptitle("PCA Data", fontweight='bold')
    plt.legend()
plt.show()

#Project data into new 3D space with various eigenvectors combinations
for i in range(2):
    X_pca3D = np.dot(X_stand, eigVec_sorted[:, i:(i+3)])
    plt.figure("PCA 3D")
    ax = plt.axes(projection='3d')
    ax.set_title('PCA 3D Data')
    ax.scatter3D(X_pca3D[y == targets[0], 0], X_pca3D[y == targets[0], 1], X_pca3D[y == targets[0], 2], marker='*', label = target_names[targets[0]])
    ax.scatter3D(X_pca3D[y == targets[1], 0], X_pca3D[y == targets[1], 1], X_pca3D[y == targets[1], 2], marker='+', label = target_names[targets[1]])
    ax.scatter3D(X_pca3D[y == targets[2], 0], X_pca3D[y == targets[2], 1], X_pca3D[y == targets[2], 2], marker='.', label = target_names[targets[2]])
    
    ax.set_xlabel('PC'+str(i+1))                                    
    ax.set_ylabel('PC'+str(i+2))
    ax.set_zlabel('PC'+str(i+3))
    ax.legend()                                                   
    plt.show()