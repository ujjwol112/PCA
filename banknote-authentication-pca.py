#import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import Data from CSV file
data_frame = pd.read_csv('BankNote_Authentication.csv')         #Convert CSV file to dataFrame
data_values = data_frame.values                                 #Get datavalues from dataFrame
data = data_values[:,:-1]                                       #Assign attributes to data variable
class_array = data_values[:,-1]                                 #Assign target classes to class_array variable

data_nor = (data-data.mean())/data.std()                        #Normalize the data
S_dat = (1/(len(data_nor)-1))*np.dot(data_nor.T, data_nor)      #Calculate covariance matrix using formula

nor_var = [np.var(data_nor[:,0]), np.var(data_nor[:,1]),np.var(data_nor[:,2]), np.var(data_nor[:,3])]   #Calculate variances of each variable into a list

eigenvalues, eigenvector = np.linalg.eig(S_dat)                 #Eigenvalues and Eigenvector calculation
pro_var = (eigenvalues/np.sum(eigenvalues))                     #Calculation of proportion of variance using formula

#Sorting of eigenvalues and eigenvector in descending order
ind = np.argsort(eigenvalues)[::-1]                             #Get sorting indices 
eigVal_sorted = eigenvalues[ind]                                #Sort eigenvalues according to indices
eigVec_sorted = eigenvector[:, ind]                             #Sort eigenvector corresponding to indices

#Project data into new 2D space with various eigenvectors combinations
for i in range(3):
    Y_2d = np.dot(eigVec_sorted[:, i:i+2].T,data_nor.T)         #Project the data onto the new feature space
    Y_2d = Y_2d.T
    X_pca_class0 = Y_2d[class_array == 0,:]                     #Gets all the rows corresponding to class 0
    X_pca_class1 = Y_2d[class_array == 1,:]                     #Gets all the rows corresponding to class 1

    plt.figure("PCA with PC"+str(i+1)+" & PC"+str(i+2))
    plt.suptitle("PCA 2D with PC"+str(i+1)+" & PC"+str(i+2), fontweight='bold')
    plt.scatter(X_pca_class0[:,0],X_pca_class0[:,1], marker='x', label = "Genuine Note")
    plt.scatter(X_pca_class1[:,0],X_pca_class1[:,1], marker= '|', label = "Forged Note")
    plt.legend()
    plt.xlabel("PC"+str(i+1))
    plt.ylabel("PC"+str(i+2))
    plt.axis('equal')
    plt.grid(True)  
    plt.show()

#Project data into new 3D space with various eigenvectors combinations
for i in range(2):
    Y_3d = np.dot(eigVec_sorted[:, i:i+3].T,data_nor.T)
    Y_3d = Y_3d.T
    X3_pca_class0 = Y_3d[class_array == 0,:]
    X3_pca_class1 = Y_3d[class_array == 1,:]

    plt.figure("PCA 3D")
    ax = plt.axes(projection='3d')
    ax.set_title('PCA 3D with PC'+str(i+1)+', PC'+str(i+2)+' & PC'+str(i+3))
    ax.scatter3D(X3_pca_class0[:,0],X3_pca_class0[:,1],X3_pca_class0[:,2], marker='x', label = "Genuine Note")
    ax.scatter3D(X3_pca_class1[:,0],X3_pca_class1[:,1],X3_pca_class1[:,2], marker='|', label = "Forged Note")
    ax.set_xlabel('PC'+str(i+1))
    ax.set_ylabel('PC'+str(i+2))
    ax.set_zlabel('PC'+str(i+3))
    ax.legend()
    plt.show()