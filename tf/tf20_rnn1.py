import tensorflow as tf
import numpy as np

#1. 데이터
idx2char = ['e', 'h', 'i', 'l', 'o']

_data = np.array([['h', 'i', 'h', 'e', 'l', 'l', 'o']], dtype=np.str).reshape(-1, 1)
print(_data.shape) #(7, 1)
print(_data)
print(type(_data))

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray()

print("========================================")
print(_data)
print(type(_data))
print(_data.dtype)

x_data = _data[:6,] #hihell
y_data = _data[1:,]  # ihello

print("========== x ===========")
print(x_data)
print("========== y ===========")
print(y_data)
print("===========   ==========")

y_data = np.argmax(y_data, axis=1)
print(y_data)
print(y_data.shape) #텐서플로우에서는 1,6으로 해야줘야된다.

x_data = x_data.reshape(1, 6, 5)
y_data = y_data.reshape(1, 6)

sequence_length = 6
input_dim = 5
output = 5
batch_size = 1 #전체행

# X = tf.placeholder(tf.float32, (None, sequence_length , input_dim))
# Y = tf.placeholder(tf.float32, (None, sequence_length))

X = tf.compat.v1.placeholder(tf.float32, (None, sequence_length , input_dim))
Y = tf.compat.v1.placeholder(tf.int32, (None, sequence_length))

print(X) #Tensor("Placeholder:0", shape=(?, 6, 5), dtype=float32)
print(Y) #세션을 통과하지않아서 텐서의 자료형이 출력


#2. 모델 수성

#model.add(LSTM(output, input_shape=(6,5)))
# cell = tf.nn.rnn_cell.BasicLSTMCell(output
cell = tf.keras.layers.LSTMCell(output)
hypothesis, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
#dynamic_rnn 동적rnn
print(hypothesis.shape) # (None,6, 5) output이 5이다.

#3. 컴파일
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=hypothesis, targets=Y, weights=weights)
cost = tf.reduce_mean(sequence_loss)

train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, axis=2)

#3-2. 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(401):
        loss, _ = sess.run([cost, train], feed_dict={X:x_data, Y:y_data}) #볼필요가 없으면 _만 사용한다.
        result = sess.run(prediction, feed_dict={X:x_data})
        print(i,"loss :",loss, "prediction :", result, "true Y:", y_data)

        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\nPrediction str :", ''.join(result_str))







