from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3], [2,3,4],[3,4,5],[4,5,6]]) #4,3
y = array([4,5,6,7]) # (4, )
# y2 = array([[4,5,6,7]]) # (1, 4)
# y3 = array([[4], [5], [6], [7]]) # (4, 1)
                 
print("x.shape: ", x.shape) #(4, 3)
print("y.shape: ", y.shape) #(4, )

#x = x.reshape(4, 3, 1)
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)

'''              행 ,       열 ,   몇개씩 자르는지
x의 shape = (batch_size, timesteps, feature)shape 구성
input_shape = (timesteps, feature)
input_length = timesteps, input_dim = feature
'''

#2. 모델 구성
model = Sequential()
#model.add(LSTM(10, activation='relu', input_shape=(3, 1)))

model.add(LSTM(10, input_length=3, input_dim=1))
# 4행 3열의 1개씩 행은 무시하므로 (3, 1)
model.add(Dense(700))
model.add(Dense(700))
model.add(Dense(700))
model.add(Dense(700))
model.add(Dense(700))
model.add(Dense(700))
model.add(Dense(1))

model.summary()

#3.실행
model.compile(optimizer='adam', loss = 'mse',)
model.fit(x, y, epochs=50, batch_size=1)

x_predict = array([5, 6, 7])
x_predict = x_predict.reshape(1,3,1)

print(x_predict)

yhat = model.predict(x_predict)
print(yhat)
