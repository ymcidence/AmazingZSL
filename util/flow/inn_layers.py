import tensorflow as tf
import numpy as np
# from util.layer import conventional_layers as layers
from util.flow import inn_ops as inn_layers


def prior(name, y_onehot, hps):
    with tf.variable_scope(name):
        n_z = hps.top_shape[-1]

        h = tf.zeros([tf.shape(y_onehot)[0]] + hps.top_shape[:2] + [2 * n_z])
        if hps.learntop:
            h = inn_layers.conv2d_zeros('p', h, 2 * n_z)
        if hps.ycond:
            h += tf.reshape(inn_layers.linear_zeros("y_emb", y_onehot,
                                                    2 * n_z), [-1, 1, 1, 2 * n_z])

        pz = inn_layers.gaussian_diag(h[:, :, :, :n_z], h[:, :, :, n_z:])

    def logp(z1):
        objective = pz.logp(z1)
        return objective

    def sample(eps=None, eps_std=None):
        if eps is not None:
            # Already sampled eps. Don't use eps_std
            z = pz.sample2(eps)
        elif eps_std is not None:
            # Sample with given eps_std
            z = pz.sample2(pz.eps * tf.reshape(eps_std, [-1, 1, 1, 1]))
        else:
            # Sample normally
            z = pz.sample

        return z

    def eps(z1):
        return pz.get_eps(z1)

    return logp, sample, eps


def f(name, h, width, n_out=None):
    n_out = n_out or int(h.get_shape()[3])
    with tf.variable_scope(name):
        h = tf.nn.relu(inn_layers.conv2d("l_1", h, width))
        h = tf.nn.relu(inn_layers.conv2d("l_2", h, width, filter_size=[1, 1]))
        h = inn_layers.conv2d_zeros("l_last", h, n_out)
    return h


def checkpoint(z, logdet):
    zshape = inn_layers.int_shape(z)
    z = tf.reshape(z, [-1, zshape[1] * zshape[2] * zshape[3]])
    logdet = tf.reshape(logdet, [-1, 1])
    combined = tf.concat([z, logdet], axis=1)
    tf.add_to_collection('checkpoints', combined)
    logdet = combined[:, -1]
    z = tf.reshape(combined[:, :-1], [-1, zshape[1], zshape[2], zshape[3]])
    return z, logdet


def revnet2d(name, z, logdet, hps, reverse=False):
    with tf.variable_scope(name):
        if not reverse:
            for i in range(hps.depth):
                z, logdet = checkpoint(z, logdet)
                z, logdet = revnet2d_step(str(i), z, logdet, hps, reverse)
            z, logdet = checkpoint(z, logdet)
        else:
            for i in reversed(range(hps.depth)):
                z, logdet = revnet2d_step(str(i), z, logdet, hps, reverse)
    return z, logdet


# Simpler, new version

def revnet2d_step(name, z, logdet, hps, reverse):
    with tf.variable_scope(name):

        shape = inn_layers.int_shape(z)
        n_z = shape[3]
        assert n_z % 2 == 0

        if not reverse:

            z, logdet = inn_layers.actnorm("actnorm", z, logdet=logdet)

            if hps.flow_permutation == 0:
                z = inn_layers.reverse_features("reverse", z)
            elif hps.flow_permutation == 1:
                z = inn_layers.shuffle_features("shuffle", z)
            # elif hps.flow_permutation == 2:
            #     z, logdet = invertible_1x1_conv("invconv", z, logdet)
            else:
                raise Exception()

            z1 = z[:, :, :, :n_z // 2]
            z2 = z[:, :, :, n_z // 2:]

            if hps.flow_coupling == 0:
                z2 += f("f1", z1, hps.width)
            elif hps.flow_coupling == 1:
                h = f("f1", z1, hps.width, n_z)
                shift = h[:, :, :, 0::2]
                # scale = tf.exp(h[:, :, :, 1::2])
                scale = tf.nn.sigmoid(h[:, :, :, 1::2] + 2.)
                z2 += shift
                z2 *= scale
                logdet += tf.reduce_sum(tf.log(scale), axis=[1, 2, 3])
            else:
                raise Exception()

            z = tf.concat([z1, z2], 3)

        else:

            z1 = z[:, :, :, :n_z // 2]
            z2 = z[:, :, :, n_z // 2:]

            if hps.flow_coupling == 0:
                z2 -= f("f1", z1, hps.width)
            elif hps.flow_coupling == 1:
                h = f("f1", z1, hps.width, n_z)
                shift = h[:, :, :, 0::2]
                # scale = tf.exp(h[:, :, :, 1::2])
                scale = tf.nn.sigmoid(h[:, :, :, 1::2] + 2.)
                z2 /= scale
                z2 -= shift
                logdet -= tf.reduce_sum(tf.log(scale), axis=[1, 2, 3])
            else:
                raise Exception()

            z = tf.concat([z1, z2], 3)

            if hps.flow_permutation == 0:
                z = inn_layers.reverse_features("reverse", z, reverse=True)
            elif hps.flow_permutation == 1:
                z = inn_layers.shuffle_features("shuffle", z, reverse=True)
            # elif hps.flow_permutation == 2:
            #     z, logdet = invertible_1x1_conv(
            #         "invconv", z, logdet, reverse=True)
            else:
                raise Exception()

            z, logdet = inn_layers.actnorm("actnorm", z, logdet=logdet, reverse=True)

    return z, logdet





def test_prior():
    import argparse
    parser = argparse.ArgumentParser()
    hps = parser.parse_args()
    hps.top_shape = [16, 16, 12]
    hps.learntop = False
    hps.ycond = True

    y = tf.constant([1, 2, 4, 3, 1])
    y = tf.one_hot(y, 4, axis=-1)
    a, b, c = prior('hehe', y, hps)
    print('hehe')


if __name__ == '__main__':
    test_prior()
