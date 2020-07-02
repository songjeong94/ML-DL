#1. 데이터
import numpy as np
x = np.array([1,2,3,4])
y = np.array([1,2,3,4])

#2.모델구성
from keras.models import Sequential
from keras.layers import Dense


from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam, Adamax

optimizer = ['Adam', 'RMSprop', 'SGD', 'Adadelta', 'Adagrad', 'Nadam', 'Adamax']

model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(11))
model.add(Dense(1))



    
#optimizer = Adam(lr = 0.001) #[[2.4634733], [3.7274847]]
#optimizer = RMSprop(lr = 0.001)  #[1.9460123],[2.756861 ]]
#optimizer = SGD(lr = 0.001) # [2.9371033], [4.4860425]]
#optimizer = Adadelta(lr = 0.001) #[[-0.31294072],[-0.5222819 ]]
#optimizer = Adagrad(lr = 0.001) #[[0.81859446], [1.2765523 ]]

for optimize in optimizer:
    model.compile(loss='mse', optimizer=optimize, metrics=['mse'])
    model.fit(x, y, epochs=100)
    loss = model.evaluate(x,y)
    pred1 = model.predict([3,5])
    print(f"{optimize}일때의 정확도:" ,pred1)
