import tensorflow as tf
import numpy as np
from util.layer import conventional_layers as layers
from util.flow import inn_ops as inn_layers
from tensorflow.contrib.framework.python.ops import add_arg_scope


@add_arg_scope
def invertible_projection(name, z, logdet, forward=True):
    # LU-decomposed version
    shape = inn_layers.int_shape(z)
    with tf.variable_scope(name):

        dtype = 'float32'

        # Random orthogonal matrix:
        import scipy
        np_w = scipy.linalg.qr(np.random.randn(shape[-1], shape[-1]))[
            0].astype('float32')

        np_p, np_l, np_u = scipy.linalg.lu(np_w)
        np_s = np.diag(np_u)
        np_sign_s = np.sign(np_s)
        np_log_s = np.log(abs(np_s))
        np_u = np.triu(np_u, k=1)

        p = tf.get_variable("P", initializer=np_p, trainable=False)
        l = tf.get_variable("L", initializer=np_l)
        sign_s = tf.get_variable(
            "sign_S", initializer=np_sign_s, trainable=False)
        log_s = tf.get_variable("log_S", initializer=np_log_s)
        # S = tf.get_variable("S", initializer=np_s)
        u = tf.get_variable("U", initializer=np_u)

        p = tf.cast(p, dtype)
        l = tf.cast(l, dtype)
        sign_s = tf.cast(sign_s, dtype)
        log_s = tf.cast(log_s, dtype)
        u = tf.cast(u, dtype)

        w_shape = [shape[-1], shape[-1]]

        l_mask = np.tril(np.ones(w_shape, dtype=dtype), -1)
        l = l * l_mask + tf.eye(*w_shape, dtype=dtype)
        u = u * np.transpose(l_mask) + tf.diag(sign_s * tf.exp(log_s))
        w = tf.matmul(p, tf.matmul(l, u))

        u_inv = tf.matrix_inverse(u)
        l_inv = tf.matrix_inverse(l)
        p_inv = tf.matrix_inverse(p)
        w_inv = tf.matmul(u_inv, tf.matmul(l_inv, p_inv))

        w = tf.cast(w, tf.float32)
        w_inv = tf.cast(w_inv, tf.float32)
        log_s = tf.cast(log_s, tf.float32)

        if forward:

            z = z @ w
            logdet += tf.reduce_sum(log_s)

            return z, logdet
        else:

            z = z @ w_inv
            logdet -= tf.reduce_sum(log_s)

            return z, logdet


def simple_nn(name, tensor_in: tf.Tensor, middle_dim, output_dim=None):
    output_dim = output_dim or tensor_in.shape.as_list()[-1]
    with tf.variable_scope(name):
        fc_1 = layers.fc_layer('fc_1', tensor_in, middle_dim)
        # fc_1 = layers.leaky_relu(fc_1)
        fc_2 = layers.fc_layer('fc_2', fc_1, output_dim)
    return fc_2


def reverse_features(name, h, reverse=False):
    return h[:, ::-1]


class SimpleINN(object):
    def __init__(self, name, hidden_size=1024, level=1, depth=3, permute=1, nn=simple_nn):
        self.name = name
        self.hidden_size = hidden_size
        self.level = level or 1
        if self.level > 1:
            raise NotImplementedError()
        self.depth = depth or 1
        self.permute = permute
        self.nn = nn

    @add_arg_scope
    def _single_depth(self, name, tensor_in: tf.Tensor, det, forward=True):
        feat_size = tensor_in.shape.as_list()[-1]
        with tf.variable_scope(name):
            if forward:
                # 1. actnorm
                # x, det = inn_layers.actnorm('actnorm', tensor_in, logdet=det)

                # 2. permutation
                if self.permute == 1:
                    x = inn_layers.shuffle_features('shuffle', tensor_in)
                elif self.permute == 0:
                    x = reverse_features('reverse', tensor_in)
                else:
                    raise NotImplementedError()

                # 3. coupling
                x1, x2 = tf.split(x, num_or_size_splits=2, axis=1)
                h = self.nn('nn', x1, self.hidden_size, feat_size)
                shift = h[:, 0::2]
                scale = tf.nn.sigmoid(h[:, 1::2] + 2.)
                x2 = (x2 + shift) * scale
                det = det + tf.reduce_sum(tf.log(scale), axis=1)
                x = tf.concat([x1, x2], axis=1)

            else:
                # 3. coupling
                x1, x2 = tf.split(tensor_in, num_or_size_splits=2, axis=1)
                h = self.nn('nn', x1, self.hidden_size, feat_size)
                shift = h[:, 0::2]
                scale = tf.nn.sigmoid(h[:, 1::2] + 2.) + 1e-8
                x2 = x2 / scale - shift

                det = det - tf.reduce_sum(tf.log(scale), axis=1)

                x = tf.concat([x1, x2], axis=1)

                # 2. permutation
                if self.permute == 0:
                    x = reverse_features('reverse', x, reverse=True)
                elif self.permute == 1:
                    x = inn_layers.shuffle_features('shuffle', x, reverse=True)
                else:
                    raise NotImplementedError()

                # 1. actnorm
                # x, det = inn_layers.actnorm('actnorm', x, logdet=det, reverse=True)

        return x, det

    @add_arg_scope
    def _single_level(self, name, tensor_in: tf.Tensor, det, forward=True):
        with tf.variable_scope(name):
            for i in range(self.depth):
                tensor_in, det = self._single_depth('depth_{}'.format(i), tensor_in, det, forward=forward)

        return tensor_in, det

    def __call__(self, tensor_in: tf.Tensor, det, forward=True):
        return self._single_level('level_1', tensor_in, det, forward=True)


class SimplerINN(SimpleINN):
    @add_arg_scope
    def _single_depth(self, name, tensor_in: tf.Tensor, det, forward=True):
        feat_size = tensor_in.shape.as_list()[-1]
        with tf.variable_scope(name):
            if forward:
                # 1. actnorm
                # x, det = inn_layers.actnorm('actnorm', tensor_in, logdet=det)
                #
                # 2. permutation
                if self.permute == 1:
                    x = inn_layers.shuffle_features('shuffle', tensor_in)
                elif self.permute == 0:
                    x = reverse_features('reverse', tensor_in)
                else:
                    raise NotImplementedError()

                # 3. coupling
                x, det = invertible_projection('nn', x, det)

            else:
                # 3. coupling
                x, det = invertible_projection('nn', tensor_in, det, forward=False)

                # 2. permutation
                if self.permute == 0:
                    x = reverse_features('reverse', x, reverse=True)
                elif self.permute == 1:
                    x = inn_layers.shuffle_features('shuffle', x, reverse=True)
                else:
                    raise NotImplementedError()

                # 1. actnorm
                # x, det = inn_layers.actnorm('actnorm', x, logdet=det, reverse=True)

        return x, det


def test_inn_1():
    model = SimpleINN('hehe', 2)
    a = tf.constant([[2., 1.], [3.5, 0.8]], dtype=tf.float32)
    with tf.variable_scope('hehe') as scope:
        enc_a, _ = model(a, 0)
        scope.reuse_variables()
        dec_a, _ = model(enc_a, 0, forward=False)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    out_1, out_2 = sess.run([enc_a, dec_a])
    print(out_1)
    print(out_2)
    print('hehe')


def test_inn_2():
    model = SimpleINN('hehe', 2, depth=1, permute=1)
    a = tf.constant([[2., 1.], [3.5, 0.8]], dtype=tf.float32)
    with tf.variable_scope('hehe') as scope:
        enc_a, det = model(a, 0)
        scope.reuse_variables()
        dec_a, _ = model(enc_a, 0, forward=False)

    loss = tf.nn.l2_loss(dec_a - a)
    op = tf.train.GradientDescentOptimizer(.01)
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
