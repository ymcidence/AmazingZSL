import tensorflow as tf

a = tf.constant([[1, 3, 5], [2, 4, 6], [3, 7, 10], [4, 5, 27]])
ind = tf.constant([1, 0, 0])

b = tf.gather(a, ind, axis=1)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
print(sess.run(b))
print('hehe')
