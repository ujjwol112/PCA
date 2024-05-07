---

# Banknote Authentication using Principal Component Analysis (PCA)

This repository contains Python code for authenticating banknotes using Principal Component Analysis (PCA). The PCA algorithm is implemented to reduce the dimensionality of the dataset and visualize the separation of genuine and forged banknotes in 2D and 3D space.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Results](#results)

## Introduction

The authentication of banknotes is crucial to prevent fraud. This project focuses on using PCA, a popular dimensionality reduction technique, to distinguish between genuine and forged banknotes. PCA identifies the principal components of the dataset, which are orthogonal vectors that capture the maximum variance in the data. By projecting the data onto these principal components, we can visualize the separation between genuine and forged banknotes in lower-dimensional space.

## Dataset

The dataset used in this project is sourced from the UCI Machine Learning Repository. It consists of features extracted from photographic images of genuine and forged banknotes. Each data point contains four numerical attributes:

- Variance of Wavelet Transformed Image (continuous)
- Skewness of Wavelet Transformed Image (continuous)
- Kurtosis of Wavelet Transformed Image (continuous)
- Entropy of Image (continuous)

The target variable indicates whether the banknote is genuine or forged (0 for genuine, 1 for forged).

## Dependencies

To run the code, you need the following dependencies:

- Python 3.x
- NumPy
- Pandas
- Matplotlib

You can install these dependencies using pip:

```bash
pip install numpy pandas matplotlib
```

## Usage

1. Clone this repository:

```bash
git clone https://github.com/ujjwol112/banknote-authentication-pca.git
```

2. Navigate to the repository directory:

```bash
cd banknote-authentication-pca
```

3. Run the Python script:

```bash
python banknote_authentication_pca.py
```

The script will execute PCA on the provided dataset and generate visualizations of the separation between genuine and forged banknotes in both 2D and 3D space.

## Results

The script generates two types of visualizations:

1. **2D Visualization**: It plots the data points in a 2D space using different combinations of principal components (PC1, PC2, PC3). Genuine and forged banknotes are represented by different markers, allowing for easy differentiation.

2. **3D Visualization**: It plots the data points in a 3D space using combinations of three principal components (PC1, PC2, PC3). This provides a more comprehensive view of the data distribution, enabling better understanding of the separation between genuine and forged banknotes.


---
