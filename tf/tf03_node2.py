import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)
node4 = tf.subtract(node1, node2)
node5 = tf.multiply(node1, node2)
node6 = tf.divide(node1, node2)

print("node1: ", node1, "node2: ", node2)
print("node3: ", node3)
print("node4: ", node4)
print("node5: ", node5)
print("node6: ", node6)

sess = tf.Session()
print("sess.run(+): ", sess.run(node3))
print("sess.run(-): ", sess.run(node4))
print("sess.run(*): ", sess.run(node5))
print("sess.run(/): ", sess.run(node6))

# 3 + 4 + 5
# 4 - 3
# 3 * 4
# 4 / 2