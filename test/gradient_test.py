import tensorflow as tf

x = tf.Variable([[1., 4], [3., 29.5]])
z = tf.matrix_determinant(x)
sess = tf.Session()
xgrad = tf.gradients(z, x)
sess.run(tf.initialize_all_variables())
print(sess.run(xgrad))
print(x.shape)