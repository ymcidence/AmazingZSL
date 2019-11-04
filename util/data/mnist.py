import scipy.io as sio
import tensorflow as tf
import os
import scipy.misc as misc
import numpy as np
import torch as th
from util.data.array_reader import ArrayReader
# noinspection PyUnresolvedReferences
import torchvision.transforms as trans

abspath = os.path.abspath(__file__)


def process_mat(file_name):
    feat_mat = sio.loadmat(file_name)
    feat = feat_mat['image_data']
    label = feat_mat['image_label']
    data_dict = {'feat': feat,
                 'label': label}
    return data_dict


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def convert_tfrecord(data, part_name):
    data_length = data['feat'].shape[0]
    save_path = os.path.join('./data', 'MNIST')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_name = os.path.join(save_path, part_name + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(file_name)

    for i in range(data_length):
        print(i)
        this_feat = _float_feature(data['feat'][i, :])
        this_label = _float_feature(data['label'][i, :])
        feat_dict = {'feat': this_feat,
                     'label': this_label}

        feature = tf.train.Features(feature=feat_dict)
        example = tf.train.Example(features=feature)
        writer.write(example.SerializeToString())

    writer.close()


def build_dataset(root_folder):
    training_file = os.path.join(root_folder, 'train_mnist.mat')
    training_dict = process_mat(training_file)

    test_file = os.path.join(root_folder, 'test_mnist.mat')
    test_dict = process_mat(test_file)

    convert_tfrecord(training_dict, 'train')
    convert_tfrecord(test_dict, 'test')


class MNISTRecord(object):
    def __init__(self, batch_size, part_name='train', sess=tf.Session()):
        self.batch_size = batch_size
        self.part_name = part_name
        self.set_size = 60000 if part_name == 'train' else 10000
        self.sess = sess
        self.data = self._load_data()
        self._iterator = self.data.make_one_shot_iterator()
        self._handle = self.sess.run(self.iterator.string_handle())
        self._output_types = self.data.output_types
        self._output_shapes = self.data.output_shapes

    def _load_data(self):
        def data_parser(tf_example: tf.train.Example):
            feat_dict = {'feat': tf.FixedLenFeature([784], tf.float32),
                         'label': tf.FixedLenFeature([10], tf.float32)}
            features = tf.parse_single_example(tf_example, features=feat_dict)
            _feat = tf.cast(features['feat'], tf.float32)
            _label = tf.cast(features['label'], tf.float32)
            return _feat, _label

        record_name = os.path.join(abspath, '../../../data/MNIST', self.part_name + '.tfrecords')
        data = tf.data.TFRecordDataset(record_name).map(data_parser, num_parallel_calls=50).prefetch(self.batch_size)
        data = data.cache().repeat().shuffle(self.set_size).batch(self.batch_size)
        return data

    @property
    def iterator(self):
        return self._iterator

    @property
    def handle(self):
        return self._handle

    @property
    def output_types(self):
        return self._output_types

    @property
    def output_shapes(self):
        return self._output_shapes


class Dataset(object):
    def __init__(self, **kwargs):
        self.sess = kwargs.get('sess', tf.Session())
        self.batch_size = kwargs.get('batch_size', 256)
        self._load_data()
        self.train_test_handle = tf.placeholder(tf.string, [])
        self.feed = self._build_feed_op()

    def _load_data(self):
        # 1. training data
        settings = {'sess': self.sess,
                    'batch_size': self.batch_size,
                    'part_name': 'train'}
        self.training_data = MNISTRecord(**settings)

        # 2. test data
        settings['part_name'] = 'test'
        self.test_data = MNISTRecord(**settings)

    def _build_feed_op(self):
        dummy_iterator = tf.data.Iterator.from_string_handle(self.train_test_handle,
                                                             output_types=self.training_data.output_types,
                                                             output_shapes=self.training_data.output_shapes)
        return dummy_iterator.get_next()

    @property
    def training_handle(self):
        return self.training_data.handle

    @property
    def test_handle(self):
        return self.test_data.handle


class MNISTArrayReader(ArrayReader):
    def _build_data(self):
        return Dataset(sess=self.sess, batch_size=self.batch_size)

    def get_batch_tensor(self, part='training'):
        batch = self.get_batch(part)
        x = (th.tensor(batch[0], dtype=th.float32).cuda() - 0.128) / 0.305
        l = th.tensor(batch[1], dtype=th.float32).cuda()
        return x, l

    @staticmethod
    def augmentation(x):
        x = x + 0.08 * th.randn_like(x)
        return x


def test_1():
    build_dataset('/home/ymcidence/Workspace/CodeGeass/MatlabWorkspace')


def test_2():
    sess = tf.Session()
    batch_size = 100
    data = Dataset(sess=sess, batch_size=batch_size)

    feed_dict = {data.train_test_handle: data.training_handle}
    a = sess.run(data.feed[0], feed_dict)
    image = a[0][0, :]
    image = np.reshape(image, [28, 28]) * 255
    file_name = str(np.argmax(a[1][0, :])) + '.jpg'
    misc.imsave('./result/' + file_name, image)

    print(image)


def test_3():
    reader = MNISTArrayReader()
    batch = reader.get_batch('training')
    print(batch)


if __name__ == '__main__':
    test_3()
