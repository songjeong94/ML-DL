# 100번을 카피해서 lr을 넣고 튠하시오

from keras.utils import np_utils
from keras.models import Model, Sequential
from keras.layers import Input, Dropout, Conv2D, Flatten
from keras.layers import MaxPooling2D, Dense,LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from keras.datasets import mnist
import numpy as np

#1.데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(y_train.shape)


x_train = x_train.reshape(-1,28)
outputs = y_train.shape
x_test = x_test.reshape(-1,28)
print(x_train.shape)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


#2.모델

def build_model(optimizer = 'adam',lr=0.1):
    model = Sequential()
    model.add(Dense(10,input_dim=28))
    model.add(Dense(100))
    model.add(Dense(1))
    
    model.compile(optimizer=optimizer, metrics=['acc'], loss = 'categorical_crossentropy')
    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    return{"batch_size": batches, "optimizer": optimizers}

model = KerasClassifier(build_fn = build_model, verbose=1)

hyperparameter = create_hyperparameters()

search = RandomizedSearchCV(model, hyperparameter, cv=3)
search.fit(x_train, y_train)


print(search.best_params_)

#y_pred = search.predict(x_test)
#y_pred = np.argmax(y_pred, axis=1)
acc = search.score(x_test, y_test)
#print("최종 정답률 : ", accuracy_score(y_test, y_pred))
print("acc: ",acc)