import tensorflow as tf
from util.layer.conventional_layers import label_select

tf.enable_eager_execution()

a = tf.one_hot(tf.random.uniform([20], minval=0, maxval=5, dtype=tf.int32), depth=6)

b = tf.constant([2, 4, 5])
b = tf.expand_dims(tf.reduce_sum(tf.one_hot(b, depth=6), 0), -1)
c = tf.matmul(a, b, transpose_b=False)
d = tf.where(c > 0)
print(tf.gather(a, d[:, 0]))
print(label_select(a, a, tf.constant([2, 4, 5]), 6))
print('hehe')
