import tensorflow as tf
from util.layer import conventional_layers as layers
from util.flow import inn_ops as inn_layers


class SingleINN(object):
    def __init__(self, name, hidden_size=1024, level=1, depth=1, permute=1):
        self.name = name
        self.hidden_size = hidden_size
        self.level = level or 1
        if self.level > 1:
            raise NotImplementedError()
        self.depth = depth or 1
        self.permute = permute

    @staticmethod
    def _simple_nn(name, tensor_in: tf.Tensor, middle_dim, output_dim=None):
        output_dim = output_dim or tensor_in.shape.as_list()[-1]
        with tf.variable_scope(name):
            fc_1 = layers.fc_relu_layer('fc_1', tensor_in, middle_dim)
            fc_2 = layers.fc_layer('fc_2', fc_1, output_dim)
        return fc_2

    def _single_depth(self, name, tensor_in: tf.Tensor, det, forward=True):
        feat_size = tensor_in.shape.as_list()[-1]
        with tf.variable_scope(name):
            if forward:
                # 1. actnorm
                x, det = inn_layers.actnorm('actnorm', tensor_in, logdet=det)

                # 2. permutation
                if self.permute == 1:
                    x = inn_layers.shuffle_features('shuffle', x)
                elif self.permute == 0:
                    x = inn_layers.reverse_features('reverse', x)
                else:
                    raise NotImplementedError()

                # 3. coupling
                x1, x2 = tf.split(x, num_or_size_splits=2, axis=1)
                h = self._simple_nn('nn', x1, self.hidden_size, feat_size)
                shift = h[:, 0::2]
                scale = tf.nn.sigmoid(h[:, 1::2] + 2.)
                x2 = (x2 + shift) * scale
                det = det + tf.reduce_sum(tf.log(scale), axis=1)
                x = tf.concat([x1, x2], axis=1)

            else:
                # 3. coupling
                x1, x2 = tf.split(tensor_in, num_or_size_splits=2, axis=1)
                h = self._simple_nn('nn', x1, self.hidden_size, feat_size)
                shift = h[:, 0::2]
                scale = tf.nn.sigmoid(h[:, 1::2] + 2.)
                x2 = x2 / scale - shift

                det = det - tf.reduce_sum(tf.log(scale), axis=1)

                x = tf.concat([x1, x2], axis=1)

                # 2. permutation
                if self.permute == 0:
                    x = inn_layers.reverse_features('reverse', x, reverse=True)
                elif self.permute == 1:
                    x = inn_layers.shuffle_features('shuffle', x, reverse=True)
                else:
                    raise NotImplementedError()

                # 1. actnorm
                x, det = inn_layers.actnorm('actnorm', x, logdet=det, reverse=True)

        return x, det

    def _single_level(self, name, tensor_in: tf.Tensor, det, forward=True):
        with tf.variable_scope(name):
            for i in range(self.depth):
                tensor_in, det = self._single_depth('depth_{}'.format(i), tensor_in, det, forward=forward)

        return tensor_in, det

    def __call__(self, tensor_in: tf.Tensor, det, forward=True):
        return self._single_level('level_1', tensor_in, det, forward=True)


def test_inn_1():
    model = SingleINN('hehe', 2)
    a = tf.constant([[2., 1.], [3.5, 0.8]], dtype=tf.float32)
    with tf.variable_scope('hehe') as scope:
        enc_a, _ = model(a, 0)
        scope.reuse_variables()
        dec_a = model(enc_a, 0, forward=False)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run(dec_a))
    print('hehe')


def test_inn_2():
    model = SingleINN('hehe', 2, depth=2)
    a = tf.constant([[2., 1.], [3.5, 0.8]], dtype=tf.float32)
    with tf.variable_scope('hehe') as scope:
        enc_a, det = model(a, 0)
        scope.reuse_variables()
        dec_a, _ = model(enc_a, 0, forward=False)

    loss = tf.nn.l2_loss(enc_a) - tf.reduce_sum(tf.abs(det))
    op = tf.train.GradientDescentOptimizer(.1)
    opt = op.minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        hehe, _, dis, loss_value = sess.run([dec_a, opt, tf.nn.l2_loss(a - dec_a), loss])

        print('Step {}, loss {}, dis {}'.format(i, loss_value, dis))
        if i % 5 == 0:
            print(hehe)


if __name__ == '__main__':
    test_inn_2()
