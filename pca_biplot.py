# ilham - 馬希迪
# M10118033
# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.pyplot import figure

# Load the Dataset
from sklearn.datasets import load_boston

boston_dataset = load_boston() 
# converting bunch data from sklean into pandas DataFrame
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

boston.head()

# the original dimension shape before PCA process
boston.shape

# Standardize the data
from sklearn.preprocessing import StandardScaler

x = StandardScaler().fit_transform(boston)
x = pd.DataFrame(x, columns=boston_dataset.feature_names)
## Standardization
# - PCA is affected by scales and different features might have different scales
# - Standardization dataset scale is necessary as it removes the biases in the 
#   original variables.
# - The standardized variables will be unitless and have similar variance
# - Standardization is an advisable method for data transformation when the variable
#   in the original dataset have been measured on a significantly different scale


# make pca model
from sklearn.decomposition import PCA

pcm5 = PCA(n_components=5)
pca1 = pcm5.fit_transform(x)

pcm2 = PCA(n_components=2)
pca2 = pcm2.fit_transform(x)
# we make two model for comparison purpose

pca1.shape
# we can see that the shape after performing fit_transform in pca1 and pca2
# the dimension of the data from (506, 13) change into (506, 5) and (506, 2)

pca2.shape

print('pca variance with 5 PCs: ',pcm5.explained_variance_)
print('pca variance ratio with 5 PCs: ',pcm5.explained_variance_ratio_)

print('pca variance with 2 PCs: ',pcm2.explained_variance_)
print('pca variance ratio with 2 PCs: ',pcm2.explained_variance_ratio_)
## Explained_variance and Eplained_variance_ratio
# - Explained_variance: is the diagonal of actual eigenvalues
# - Explained_variance_ration: is explained variance in PCs like,
#   in PC1 is about 47% and in the PC2 is about 11% of explained variance


# Scree plot with 5 PCs
figure(figsize=(8, 6), dpi=80)
plt.plot(pcm5.explained_variance_)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance / eigenvalue')
plt.show()
## Scree plot
# from the graph we can see after the Principal Component 2
# the eigenvalue or explained variance has decreasing and become more
# flatten and in PCA we use to keep the PCs that greater than 1 
# so we can keep the PC1 and PC2


# Scree plot with 2 PCs
figure(figsize=(8, 6), dpi=80)
plt.plot(pcm2.explained_variance_)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance / eigenvalue')
plt.show()


# plotting Scatter plot or Score plot
figure(figsize=(8, 6), dpi=80)
plt.scatter(pca1[:, 0], pca1[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


# PCA Biplot
def myplot(score,coeff,labels=None):
    figure(figsize=(8, 6), dpi=80)
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley,s=5)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1], color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'green', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
 
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

myplot(pca1[:,0:2],np.transpose(pcm5.components_[0:2, :]),list(x.columns))
plt.show()

## Biplot is actually a plot who has contain two plot which is,
# 1.Scatter plot of PC1 and PC2 or the other name is Score Plot
# 2.Loading plot which show how strong each caracteristic influences a principal component

## Scatter plot
# - the Scatter plot show that PCs describe variation and account for the varied influences
#   of the original characteristics. 

## Loading plot
# - the angels between the vectors tell us how characteristics correlated with one another
# - when two vectors are close, forming a small angle, the two variables they represent are 
#   positively correlated like, B and RM
# - when they forming an 90°, they don't have any correlation like, DIS and CRIM
# - when they diverge and form a large angle (close to 180°), they are negative corellated 
#   like,DIS and AGE





