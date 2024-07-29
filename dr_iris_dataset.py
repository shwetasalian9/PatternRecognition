#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
The Iris Dataset
=========================================================
This data sets consists of 3 different types of irises'
(Setosa, Versicolour, and Virginica) petal and sepal
length, stored in a 150x4 numpy.ndarray

The rows being the samples and the columns being:
Sepal Length, Sepal Width, Petal Length, and Petal Width.

The below plot uses the first two features.
See `here <https://en.wikipedia.org/wiki/Iris_flower_data_set>`_ for more
information on this dataset.
"""
print(__doc__)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding

def PCAProjection(myArray, dim):
    return PCA(n_components=dim).fit_transform(myArray)

def LLEProjection(myArray, dim):
    embedding = LocallyLinearEmbedding(n_components=dim)
    X_transformed = embedding.fit_transform(myArray)
    return X_transformed 

def ISOMAPProjection(myArray, dim):
    embedding = Isomap(n_components=dim)
    X_transformed = embedding.fit_transform(myArray)
    return X_transformed

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# To get a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(figsize=(15, 5))

# PCA - 3D
ax1 = fig.add_subplot(131, projection='3d')
X_reduced_pca = PCAProjection(iris.data, dim=3)
ax1.scatter(X_reduced_pca[:, 0], X_reduced_pca[:, 1], X_reduced_pca[:, 2], c=y, cmap=plt.cm.Set1, edgecolor='k', s=40)
ax1.set_title("PCA - 3D")
ax1.set_xlabel("1st eigenvector")
ax1.set_ylabel("2nd eigenvector")
ax1.set_zlabel("3rd eigenvector")

# Isomap - 3D
ax2 = fig.add_subplot(132, projection='3d')
X_reduced_isomap = ISOMAPProjection(iris.data, dim=3)
ax2.scatter(X_reduced_isomap[:, 0], X_reduced_isomap[:, 1], X_reduced_isomap[:, 2], c=y, cmap=plt.cm.Set1, edgecolor='k', s=40)
ax2.set_title("Isomap - 3D")
ax2.set_xlabel("1st eigenvector")
ax2.set_ylabel("2nd eigenvector")
ax2.set_zlabel("3rd eigenvector")

# LLE - 3D
ax3 = fig.add_subplot(133, projection='3d')
X_reduced_lle = LLEProjection(iris.data, dim=3)
ax3.scatter(X_reduced_lle[:, 0], X_reduced_lle[:, 1], X_reduced_lle[:, 2], c=y, cmap=plt.cm.Set1, edgecolor='k', s=40)
ax3.set_title("Locally Linear Embedding - 3D")
ax3.set_xlabel("1st eigenvector")
ax3.set_ylabel("2nd eigenvector")
ax3.set_zlabel("3rd eigenvector")

plt.show()

# Separate figure for 2D visualizations
fig2D = plt.figure(figsize=(15, 5))

# PCA - 2D
ax4 = fig2D.add_subplot(131)
X_reduced_pca_2d = PCAProjection(iris.data, dim=2)
ax4.scatter(X_reduced_pca_2d[:, 0], X_reduced_pca_2d[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k', s=40)
ax4.set_title("PCA - 2D")
ax4.set_xlabel("1st eigenvector")
ax4.set_ylabel("2nd eigenvector")

# Isomap - 2D
ax5 = fig2D.add_subplot(132)
X_reduced_isomap_2d = ISOMAPProjection(iris.data, dim=2)
ax5.scatter(X_reduced_isomap_2d[:, 0], X_reduced_isomap_2d[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k', s=40)
ax5.set_title("Isomap - 2D")
ax5.set_xlabel("1st eigenvector")
ax5.set_ylabel("2nd eigenvector")

# LLE - 2D
ax6 = fig2D.add_subplot(133)
X_reduced_lle_2d = LLEProjection(iris.data, dim=2)
ax6.scatter(X_reduced_lle_2d[:, 0], X_reduced_lle_2d[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k', s=40)
ax6.set_title("Locally Linear Embedding - 2D")
ax6.set_xlabel("1st eigenvector")
ax6.set_ylabel("2nd eigenvector")

plt.show()
