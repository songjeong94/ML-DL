import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

ds = load_diabetes()

X = ds.data
Y = ds.target

pca = PCA(n_components= 5)
x2 = pca.fit_transform((X))
pca_evr = pca.explained_variance_ratio_
print(pca_evr)
print(sum(pca_evr))

