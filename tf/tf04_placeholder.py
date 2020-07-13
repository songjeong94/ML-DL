
#tensorflow는 노드 연산을 한다.

#tensorflow에서는 graph의 연산에게 직접 tensor 값을 줄 수 있는 
# feed 메커니즘도 제공한다.


import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)    # 변하지 않는 값
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2) 

sess = tf.Session() # 강의장 준비

a = tf.placeholder(tf.float32)     # , 강의장 예약,input와 비슷한 개념 
b = tf.placeholder(tf.float32)     # : sess.run할 때 fedd_dict로 값을 집어넣어준다.

adder_node = a + b #강의장 의자 배치
                                                        # feead bit 집어넣을 값들
print(sess.run(adder_node, feed_dict = {a:3, b:4.5}))   # dict안에 들어간 수를 빼와서 사용
print(sess.run(adder_node, feed_dict = {a:[1, 3], b:[2, 4]})) # [3. 7.]

add_and_triple = adder_node * 3                         # (a + b) *3
print(sess.run(add_and_triple, feed_dict={a:3, b:4.5})) # 22.5
