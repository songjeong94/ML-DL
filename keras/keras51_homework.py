
#x = [1, 2, 3]
#x = x - 1
#print(x)
import numpy as np

y = np.array([1,2,3,4,5,1,2,3,4,5])
y = y - 1

from keras.utils import np_utils

y = np_utils.to_categorical(y)
print(y)
print(y.shape)

#2번의 두 번째 답


y= np.array([1,2,3,4,5,1,2,3,4,5,])
print(y.shape)
y = y.reshape(10, 1)
from sklearn.preprocessing import OneHotEncoder
aaa  = OneHotEncoder()
aaa.fit(y) 
y = aaa.transform(y).toarray()

print(y)
print(y.shape)
