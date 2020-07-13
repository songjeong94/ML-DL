import tensorflow as tf
tf.set_random_seed(777) # randomseed로 고정

x_train = [1, 2, 3]
y_train = [3, 5, 7]

#변수// 0~1사이의 정규확률 분포 값을 생성
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#sess = tf.Session()
## 변수는 항상 초기화 시키고 작업해야한다.
#sess.run(tf.global_variables_initializer())
#print(sess.run(W))

hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train)) # MSE
#전체 평균을 구한다 모든차원이 제거되고 하나의 스칼라 값 출력
# cost = loss
# reduce_mean = 평균

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost) #optimizer //최소의 loss가 최적의  weight값을 구한다.

#Session범위 안에 다 들어감
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer) #전체 변수 초기화
    ## sess 범주 안에 있는 것들, 원래 session을 열면 다시 닫아야 하지만 (  sess.close() ) ,
    # 이렇게 with 문을 사용하면 안닫아도 된다 메소드가  __enter__ 및 __exit__으로 이루어져있기 때문
    # variable 변수는 항상 초기화를 시켜주어야 한다, 변수가 들어갈 메모리 할당
    # 참고로 tf.variable은 default로 'trainable=True' 라서 그래프 연산할 때 자동으로 학습된다 


    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b])
        #train은 하되 결과값은 출력하지 않겠다.//cost를 minimize

        if step % 20==0: #20번마다 출력
            print(step, cost_val, W_val, b_val)
#session을 열었으면 닫아주어야 함
