# iris를 케라스 파이프라인 구성
# randomizedSearchCV


import numpy as np
from sklearn.datasets import load_iris
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten, Dense
from keras.layers import MaxPooling2D
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

dataset = load_iris()

x = dataset.data
y = dataset.target

# print(x.shape)    # (150, 4)
# print(y.shape)    # (150, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, 
                                        shuffle = True, random_state = 255)


# parameter = [
#         {"rf__n_estimators": [2, 5, 10]},
#         {"rf__max_depth": [1,10,100,1000]},
#         {"rf__min_samples_leaf": [1,10,100,1000]}]


# 2. 모델

def build_model(drop = 0.5, optimizer = 'adam'):
    inputs = Input(shape = (4, ), name = 'input')
    x = Dense(512, activation = 'relu', name = 'h1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation = 'relu', name = 'h2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation = 'relu', name = 'h3')(x)
    x = Dropout(drop)(x)
    x = Dense(64, activation = 'relu', name = 'h4')(x)
    x = Dropout(drop)(x)
    output = Dense(3, activation = 'softmax', name = 'output')(x)
    
    model = Model(inputs = inputs, outputs = output)
    model.compile(optimizer = optimizer, metrics = ['acc'], loss = 'categorical_crossentropy')
    return model

def create_hyperParameter():
    batches = [32, 64, 128, 256]
    # optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    epochs = [20, 40 ,60 ,80]
    return{"rf__batch_size" : batches, "rf__epochs": epochs}

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
# 케라스 분류 모델을 사이킷런 형태로 싸겠습니다.

model = KerasClassifier(build_fn = build_model, verbose = 1)
hyperparameter = create_hyperParameter()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

pipe = Pipeline([("scaler", MinMaxScaler()),('rf', model)])

search = RandomizedSearchCV(pipe, hyperparameter, cv = 3)
search.fit(x_train, y_train)

acc = search.score(x_test, y_test)
print(search.best_params_)
# print(search.best_estimator_)

# y_pred = model.predict(x_test)
print("acc" , acc)
# print("최종 정답률: ", accuracy_score(y_test, y_pred))
