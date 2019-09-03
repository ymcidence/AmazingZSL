import tensorflow as tf

SCALE = 1.


def distance(tensor_a: tf.Tensor, tensor_b: tf.Tensor):
    """
    pair-wise distances of two row-ordered matrices

    :param tensor_a: [N D]
    :param tensor_b: [M D]
    :return: a matrix of [N M]
    """

    sum_a = tf.reduce_sum(tf.square(tensor_a), 1)
    sum_b = tf.reduce_sum(tf.square(tensor_b), 1)

    sum_a = tf.reshape(sum_a, [-1, 1])
    sum_b = tf.reshape(sum_b, [1, -1])

    mul = tf.matmul(tensor_a, tensor_b, transpose_a=False, transpose_b=True)

    distances = sum_a - 2 * mul + sum_b

    return distances


# noinspection PyUnusedLocal
def basic_mmd(tensor_a: tf.Tensor, tensor_b: tf.Tensor, kernel='IMQ', scale=SCALE):
    """

        :param tensor_a: [N D] features of some data
        :param tensor_b: [M D] features of some other data
        :param kernel: only 'IMQ' supported
        :param scale: kernel hyper-parameter
        :return:
    """
    dist_1 = distance(tensor_a, tensor_a)
    dist_2 = distance(tensor_b, tensor_b)
    dist_3 = distance(tensor_a, tensor_b)

    batch_size = tf.cast(tf.shape(tensor_a)[0], dtype=tf.float32)
    s = tensor_a.shape.as_list()[1]
    c = 20

    if kernel == 'IMQ':
        kernelized_1 = c / (c + dist_1)
        kernelized_1 = tf.reduce_sum(kernelized_1) / (batch_size * (batch_size - 1))

        kernelized_2 = c / (c + dist_2)
        kernelized_2 = tf.reduce_sum(kernelized_2) / (batch_size * (batch_size - 1))

        kernelized_3 = c / (c + dist_3)
        kernelized_3 = 2 * tf.reduce_sum(kernelized_3) / (batch_size * batch_size)

        return kernelized_1 + kernelized_2 - kernelized_3
    else:
        raise Exception("I haven't considered other kernels!")


def category_mmd(tensor_a: tf.Tensor, tensor_b: tf.Tensor, label_a: tf.Tensor, label_b: tf.Tensor, kernel='RBF',
                 scale=SCALE):
    """

    :param tensor_a: [N D] features of labeled data
    :param tensor_b: [M D] features of unlabeled data
    :param label_a: one-hot labels
    :param label_b: gumbel reps of unlabeled cls prob
    :param kernel: 'RBF' or 'IMQ'
    :param scale: kernel hyper-parameter
    :return:
    """
    # 1. labeled part
    dist_1 = distance(tensor_a, tensor_a)
    mask_1 = tf.matmul(label_a, label_a, transpose_b=True)
    batch_size_1 = tf.cast(tf.shape(tensor_a)[0], dtype=tf.int32)
    mask_1 = mask_1 * (1. - tf.eye(batch_size_1))

    # 2. unlabeled part
    dist_2 = distance(tensor_b, tensor_b)
    mask_2 = tf.matmul(label_b, label_b, transpose_b=True)
    batch_size_2 = tf.cast(tf.shape(tensor_b)[0], dtype=tf.int32)
    mask_2 = mask_2 * (1. - tf.eye(batch_size_2))

    # 3. mixed discrepancy
    dist_3 = distance(tensor_a, tensor_b)
    mask_3 = tf.matmul(label_a, label_b, transpose_b=True)

    rslt = 0.

    if kernel == 'IMQ':
        for s in [.1, .2, .5, 1., 2., 5., 10.]:
            c = 2 * s * scale
            kernelized_1 = mask_1 * c / (c + dist_1)
            kernelized_1 = tf.reduce_mean(kernelized_1)

            kernelized_2 = mask_2 * c / (c + dist_2)
            kernelized_2 = tf.reduce_mean(kernelized_2)

            kernelized_3 = mask_3 * c / (c + dist_3)
            kernelized_3 = tf.reduce_mean(kernelized_3)

            rslt += kernelized_1 + kernelized_2 - 2 * kernelized_3

    else:
        kernelized_1 = mask_1 * tf.exp(- dist_1 / 2. / scale)
        kernelized_1 = tf.reduce_mean(kernelized_1)

        kernelized_2 = mask_1 * tf.exp(- dist_2 / 2. / scale)
        kernelized_2 = tf.reduce_mean(kernelized_2)

        kernelized_3 = mask_1 * tf.exp(- dist_3 / 2. / scale)
        kernelized_3 = tf.reduce_mean(kernelized_3)

        rslt += kernelized_1 + kernelized_2 - 2 * kernelized_3

    return rslt
