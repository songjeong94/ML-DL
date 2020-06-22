from keras.models import Sequential #순차적 케라스 모델 
from keras.layers import Dense # 케라스에서는 전결합층을 Dense 클래스로 구현된다. 
# Dense 레이어는 입력 뉴런 수에 상관없이 출력 뉴런 수를 자유롭게 설정할 수 있기 때문에 출력층으로 많이 사용된다.
import numpy as np

x_train = np.array([1,2,3,4,5,6,7,8,9,10]) # x항 훈련 데이터
y_train = np.array([1,2,3,4,5,6,7,8,9,10]) # y항 훈련 데이터
x_test = np.array([101,102,103,104,105,106,107,108,109,110]) # x항 테스트 데이터
y_test = np.array([101,102,103,104,105,106,107,108,109,110]) # y항 테스트 데이터

model = Sequential() #모델 구성 순차적
model.add(Dense(5, input_dim =1, activation='relu')) 
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1, activation='relu'))

model.summary() #노드의 개수와 파라미터 확인 최종파라미터는 훈련횟수

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100,batch_size=1, validation_data=(x_train,y_train)) #모델훈련 구성
loss, acc=model.evaluate(x_test, y_test, batch_size=1)

print("loss :",loss)
print("acc:", acc)

output = model.predict(x_test)
print("결과물: \n", output)
