import tensorflow as tf
import numpy as np


def invertible_projection(name, tensor_in: tf.Tensor, log_det, forward=True, reuse=None):
    """
    Invertible fully-connected layer
    :param name:
    :param tensor_in:
    :param log_det:
    :param forward:
    :param reuse:
    :return:
    """
    with tf.variable_scope(name, reuse=reuse):
        channel_num = tensor_in.shape.as_list()
        w_init = np.linalg.qr(np.random.randn(channel_num, channel_num))[0]
        w = tf.get_variable("weights", initializer=w_init)
        this_log_det = tf.log(abs(tf.matrix_determinant(w)))

        if forward:
            tensor_out = tf.matmul(tensor_in, w)
            log_det = log_det + this_log_det
        else:
            inv = tf.matrix_inverse(w)
            tensor_out = tf.matmul(tensor_in, inv)
            log_det = log_det - this_log_det

    return tensor_out, log_det


def inn(name, tensor_a: tf.Tensor, tensor_b: tf.Tensor, transform: invertible_projection):
    """
    Invertible NN
    :param name:
    :param tensor_a:
    :param tensor_b:
    :param transform: a certain function
    :return:
    """
    # 1.forward
    predicted_b, det_fw = transform(name, tensor_a, 0)
    # 2.reverse
    predicted_a, det_rev = transform(name, tensor_b, 0, forward=False, reuse=True)
    return predicted_b, det_fw, predicted_a, det_rev
