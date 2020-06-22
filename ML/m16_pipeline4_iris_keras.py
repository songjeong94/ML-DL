from sklearn.datasets import load_iris 
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input, Dropout, Flatten,Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import warnings

#1.데이터
iris = load_iris()

x = iris.data
y = iris.target

print(x.shape)
print(y.shape)


x_train,x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2)

# x_train = x_train.reshape(x_train.shape[0] , 4, 1)
# x_test = x_test.reshape(x_test.shape[0], 4, 1)


#2.모델

def build_model(drop=0.5, optimizer = 'adam'):
    inputs = Input(shape=(4,), name = 'input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(64, activation='relu', name='hidden4')(x)
    outputs = Dense(3, activation='softmax', name = 'output')(x)

    model = Model(inputs = inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],
                loss = 'categorical_crossentropy')
    return model

def create_hyperparameters():
    batches = [10, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3, 0.4, 0.5] #리스트형태로도 지정가능[0.1, 0.2, 0.3, 0.4, 0.5]
    return{"kerasclassifier__batch_size": batches, "kerasclassifier__optimizer": optimizers,
            "kerasclassifier__drop" : dropout}

model = KerasClassifier(build_fn = build_model, verbose=1)

hyperparameter = create_hyperparameters()

pipe = Pipeline([("scaler", MinMaxScaler()),('rf', model)])
#pipe = make_pipeline(MinMaxScaler(), KerasClassifier(build_fn = build_model, verbose=1))

search = RandomizedSearchCV(pipe, hyperparameter, cv = 5)
search.fit(x_train, y_train)

acc = search.score(x_test, y_test)
print(search.best_params_)
# print(search.best_estimator_)

# y_pred = model.predict(x_test)
print("acc" , acc)
# print("최종 정답률: ", accuracy_score(y_test, y_pred))
