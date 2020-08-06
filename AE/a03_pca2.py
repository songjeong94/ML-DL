import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes
from tensorflow.keras.datasets import cifar10

# ds = load_diabetes()
# X = ds.data
# Y = ds.target
# print(X.shape)

train_set, test_set = cifar10.load_data()
x_train, y_train = train_set
x_test, y_test = test_set
print(x_train.shape)
# pca = PCA(n_components= 5)
# x2 = pca.fit_transform((X))
# pca_evr = pca.explained_variance_ratio_
# print(pca_evr)
# print(sum(pca_evr))

x_train = x_train.reshape(50000, 3072)

pca = PCA()
pca.fit(x_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)

n_components = np.argmax(cumsum >= 0.90)+1
print(cumsum>=0.90)
print(n_components) # n_comp 7이상을 사용하여라

