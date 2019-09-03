import tensorflow as tf

a = tf.constant([[1, 2, 5, 11.5], [6, 4, 7, 0], [2, 2, 6, 1]], dtype=tf.float32)
w = tf.constant([[1, 7, 1], [0, 2, 2]], dtype=tf.float32)
c = w @ a
d = tf.gradients(c, a)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
print(sess.run(d))
print(sess.run(c))

# c_1 = c[:, :2]
# c_2 = c[:, 2:]
#
# d = tf.gradients(c, a)
# d_1 = tf.gradients(c_1, a)
# d_2 = tf.gradients(c_2, a)
# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
# print(sess.run(d))
# print(sess.run(d_1))
# print(sess.run(d_2))
