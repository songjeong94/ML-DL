import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import r2_score

dataset = np.array([1,2,3,4,5,6,7,8,9,10])
print(dataset.shape) # (10,)

# RNN 모델을 짜시오
size = 5 

def split_x(seq, size):
    aaa = [] # 임심 메모리 리스트
    for i in range(len(seq) - size + 1):#seq - size +1  = 값 행종료값
        subset = seq[i: (i+size)] #열 지정
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(dataset, size)
print(dataset)

x_data = dataset[:, 0:4]
print(x_data.shape) #(6,4)
print(x_data)
y_data = dataset[:, 4]
print(y_data.shape) #(6,)
print(y_data)

x_data = x_data.reshape(1, 6, 4)
y_data = y_data.reshape(6, 1)

sequence_length = 6
input_dim = 4
output = 4
batch_size = 1

X = tf.compat.v1.placeholder(tf.float32, (None, sequence_length, input_dim))
Y = tf.compat.v1.placeholder(tf.float32, (None, 1))

#2. 모델 생성
cell = tf.keras.layers.LSTMCell(output)
hypothesis, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
print(hypothesis.shape) #(?, 6, 4)

#3.컴파일
weights = tf.ones([batch_size, sequence_length])
cost =  tf.reduce_mean(tf.square( hypothesis - Y))

train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

prediction = np.array([7,8,9,10])
prediction = tf.compat.v1.placeholder(tf.float32, (None, 4, 1))

#3-2. 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(401):
        loss, _ = sess.run([cost, train], feed_dict={X:x_data, Y:y_data}) #볼필요가 없으면 _만 사용한다.
        print(i,"loss :",loss)
    y_pred =sess.run(hypothesis, feed_dict={X:x_data})
    r2 = r2_score(y_data, y_pred)
    print("R2 :", r2)

    
