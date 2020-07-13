import tensorflow as tf
tf.set_random_seed(777)

x = [1, 2, 3]
y = [3, 5 ,7]

x_train = tf.placeholder(tf.float32, shape= [None]) 
y_train = tf.placeholder(tf.float32, shape= [None])

W = tf.Variable(tf.random([1], name = 'weight')) #난수를주는 이유
b = tf.Variable(tf.random([1], name = 'bias')) # 시작 위치가 달라져도 최적값을 찾나확인하기위해

hypothesis = x_trian * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

train(tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

#with tf.Session() as sess:
with tf.compat.v1.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    sess.run(tf.compat.v1.global_variables_initializer())
    #변수에 메모리를 할당하고 초기값 설정 이렇게 쓰는것을 권한다.

    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict = {x_train:[1, 2, 3], y_train:[3, 5, 7]}) 
        # _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict = {x_train:x, y_train:y}) 

        if step % 20 == 0:
            print(step, cost_val, W_val,b_val)

    #predict 해보자
    print("예측 :",sess.run(hypothesis, feed_dict={x_train:[4]})
    print("예측 :",sess.run(hypothesis, feed_dict={x_train:5,6]})
    print("예측 :",sess.run(hypothesis, feed_dict={x_train:[6,7,8]})
    
