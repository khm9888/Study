import numpy as np

from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

dataset = load_diabetes()

x= dataset.data
y= dataset.target

print(x.shape)
print(y.shape)

pca = PCA(n_components=5)
x2=pca.fit_transform(x)
pca_evr=pca.explained_variance_ratio_
print(pca_evr)
print(sum(pca_evr))

# [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856]
# 0.8340156689459766