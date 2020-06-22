from keras.datasets import mnist 
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input, Dropout, Conv2D, Flatten
from keras.layers import MaxPooling2D, Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import numpy as np

#1.데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(y_train.shape)

x_train = x_train.reshape(x_train.shape[0] , 28, 28, 1)/255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)/255

#x_train = x_train.reshape(x_train.shape[0] , 28 * 28)/255
#x_test = x_test.reshape(x_test.shape[0], 28 * 28)/255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)

#2.모델

def build_model(optimizer = 'adam', pool_size=5):
    inputs = Input(shape=(28,28,1), name = 'input')
    x = Conv2D(100,(2,2),padding='same', name='hidden1')(inputs)
    max1 = MaxPooling2D(pool_size)(x)
    x = Conv2D(200,(2,2),padding='same', name='hidden2')(max1)
    max2 = MaxPooling2D(pool_size)(x)
    x = Conv2D(150,(2,2),padding='same', name='hidden3')(max2)
    f1 = Flatten()(x)
    outputs = Dense(10, activation='softmax', name = 'output')(f1)
    model = Model(inputs = inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],
                loss = 'categorical_crossentropy')
    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    #Maxpool = np.linspace([1,1] , [5,5], 5) #리스트형태로도 지정가능[0.1, 0.2, 0.3, 0.4, 0.5]
    return{"batch_size": batches, "optimizer": optimizers,
            }

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