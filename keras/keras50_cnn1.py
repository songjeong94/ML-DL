from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten #4차원(장수,가로,세로,명암)


model = Sequential()
model.add(Conv2D(10, (2,2), # (2,2) = 픽셀을 2 by 2 씩 잘른다.
         input_shape=(10,10,1))) #(가로,세로,명암 1=흑백, 3=칼라)(행, 열 ,채널수) # batch_size, height, width, channels
model.add(Conv2D(7,(3,3)))    #strides : 높이와 너비를 따라 컨벌루션의 보폭을 지정하는 정수 또는 튜플 / 2 개의 정수 목록입니다. 모든 공간 치수에 대해 동일한 값을 지정하는 단일 정수일 수 있습니다. 모든 보폭 값! = 1을 지정하면 모든 dilation_rate값! = 1 을 지정할 수 없습니다 .
model.add(Conv2D(5,(2,2), padding='same'))
model.add(Conv2D(5,(2,2)))
#model.add(Conv2D(5,(2,2), strides=2))
#model.add(Conv2D(5,(2,2),strides=2, padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten()) # 2차원으로 변경
model.add(Dense(1))
model.summary()

# 입력채널 * 캐널 폭* 캐널 높이 * 출력 채널 )+( bias * 출력 채널)
# 입력채널수 * 필터 폭 * 필터 높이 * 출력채널수) +( bias * 출력 채널)
# (input * kernal * kernla + bias) * output


 