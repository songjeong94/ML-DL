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

# x_train = x_train.reshape(x_train.shape[0] , 28, 28, 1)/255
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)/255

x_train = x_train.reshape(x_train.shape[0] ,784,1 )/255
x_test = x_test.reshape(x_test.shape[0], 784,1)/255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)

weight = 0.5
input = 0.5 # 시작값
goal_prediction = 0.8 #원하는 목표값
lr = 0.001

for x in x_train:
    prediction = input * weight
    error = (prediction - goal_prediction)**2

    print("Error: " + str(error) + "\tPrediction: " + str(prediction))

    up_prediction = input * (weight + lr)
    up_error = (goal_prediction - up_prediction) **2

    down_prediction = input * (weight - lr)
    down_error = (goal_prediction - down_prediction) **2

    if(down_error < up_error) :
        weight = weight - lr
    elif(down_error > up_error) :
        weight = weight + lr


#2.모델

def build_model(drop=0.5, optimizer = 'adam'):
    inputs = Input(shape=(784,1), name = 'input')
    x = LSTM(100, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name = 'output')(x)
    model = Model(inputs = inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],
                loss = 'categorical_crossentropy')
    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1 , 0.5, 5) #리스트형태로도 지정가능[0.1, 0.2, 0.3, 0.4, 0.5]
    return{"batch_size": batches, "optimizer": optimizers,
            "drop" : dropout}

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