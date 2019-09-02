import os
import numpy as np
import tensorflow as tf
from util.data import set_profiles


class ZSLMeta(object):
    def __init__(self, **kwargs):
        self.set_name = kwargs.get('set_name', 'AWA1')
        self.meta_name = os.path.join('./data', self.set_name, 'meta_info.npy')

        self.data = self._load_data()

    def _load_data(self):
        np_data = np.load(self.meta_name).item()
        # noinspection PyUnresolvedReferences
        feat_dict = {'s_cls': np_data['s_cls'].astype(np.int32),
                     'u_cls': np_data['u_cls'].astype(np.int32),
                     'cls_emb': np_data['cls_emb'].astype(np.float32),
                     's_cls_emb': np_data['s_cls_emb'].astype(np.float32),
                     'u_cls_emb': np_data['u_cls_emb'].astype(np.float32)}
        return tf.data.Dataset.from_tensors(feat_dict).repeat()

    @property
    def iterator(self):
        return self.data.make_one_shot_iterator()


class ZSLRecord(object):
    def __init__(self, **kwargs):
        self.set_name = kwargs.get('set_name', 'AWA1')
        self.part_name = kwargs.get('part_name', set_profiles.PART_NAME[0])
        self.sess = kwargs.get('sess', tf.Session())
        self.batch_size = kwargs.get('batch_size', 256)
        self.set_size = set_profiles.SET_SIZE[self.set_name][set_profiles.PART_NAME.index(self.part_name)]
        self.meta_data = kwargs.get('meta_data', ZSLMeta(set_name=self.set_name))
        self.data = self._load_data()

    def _load_data(self):
        set_name = self.set_name

        def data_parser(tf_example: tf.train.Example):
            feat_dict = {'id': tf.FixedLenFeature([], tf.int64),
                         'feat': tf.FixedLenFeature([set_profiles.FEAT_DIM[set_name]], tf.float32),
                         'label': tf.FixedLenFeature([], tf.int64),
                         'label_emb': tf.FixedLenFeature([set_profiles.ATTR_DIM[set_name]], tf.float32)}
            features = tf.parse_single_example(tf_example, features=feat_dict)

            _id = tf.cast(features['id'], tf.int32)
            _feat = tf.cast(features['feat'], tf.float32)
            _label = tf.cast(features['label'], tf.int32)
            _label_emb = tf.cast(features['label_emb'], tf.float32)
            return _id, _feat, _label, _label_emb

        record_name = os.path.join('./data', self.set_name, self.part_name + '.tfrecords')
        data = tf.data.TFRecordDataset(record_name).map(data_parser, num_parallel_calls=50).prefetch(self.batch_size)
        data = data.cache().repeat().shuffle(self.set_size).batch(self.batch_size)

        # data = data.cache().repeat().batch(self.batch_size)

        mixed_data = tf.data.Dataset.zip((data, self.meta_data.data)).map(
            lambda x, y: {'id': x[0],
                          'feat': x[1],
                          'label': x[2],
                          'label_emb': x[3],
                          's_cls': y['s_cls'],
                          'u_cls': y['u_cls'],
                          'cls_emb': y['cls_emb'],
                          's_cls_emb': y['s_cls_emb'],
                          'u_cls_emb': y['u_cls_emb']})

        return mixed_data

    @property
    def iterator(self):
        return self.data.make_one_shot_iterator()

    @property
    def handle(self):
        return self.sess.run(self.iterator.string_handle())

    @property
    def output_types(self):
        return self.data.output_types

    @property
    def output_shapes(self):
        return self.data.output_shapes


class Dataset(object):
    def __init__(self, **kwargs):
        self.set_name = kwargs.get('set_name', 'AWA1')
        self.sess = kwargs.get('sess', tf.Session())
        self.batch_size = kwargs.get('batch_size', 256)
        self._load_data()
        self.train_test_handle = tf.placeholder(tf.string, [])
        self.feed = self._build_feed_op()

    def _load_data(self):
        # 0. meta information of the dataset
        self.meta_data = ZSLMeta(set_name=self.set_name)

        # 1. training data
        settings = {'set_name': self.set_name,
                    'sess': self.sess,
                    'batch_size': self.batch_size,
                    'part_name': set_profiles.PART_NAME[0],
                    'meta_data': self.meta_data}
        self.training_data = ZSLRecord(**settings)

        # 2. seen data for test
        settings['part_name'] = set_profiles.PART_NAME[1]
        self.seen_data = ZSLRecord(**settings)

        # 3. unseen data for test
        settings['part_name'] = set_profiles.PART_NAME[2]
        self.unseen_data = ZSLRecord(**settings)

    def _build_feed_op(self):
        dummy_iterator = tf.data.Iterator.from_string_handle(self.train_test_handle,
                                                             output_types=self.training_data.output_types,
                                                             output_shapes=self.training_data.output_shapes)
        return dummy_iterator.get_next()

    @property
    def training_handle(self):
        return self.training_data.handle

    @property
    def seen_handle(self):
        return self.seen_data.handle

    @property
    def unseen_handle(self):
        return self.unseen_data.handle


# noinspection PyUnusedLocal
def test_0():
    dataset = ZSLMeta(set_name='AWA1')
    sess = tf.Session()
    feed = dataset.data.make_one_shot_iterator().get_next()
    a = sess.run(feed)
    b = sess.run(feed)


# noinspection PyUnusedLocal
def test_1():
    sess = tf.Session()
    dataset = ZSLRecord(set_name='AWA1', sess=sess, batch_size=256, part_name='train')
    feed = dataset.data.make_one_shot_iterator().get_next()
    a = sess.run(feed)
    b = sess.run(feed)


# noinspection PyUnusedLocal
def test_2():
    sess = tf.Session()
    dataset = Dataset(set_name='AWA1', sess=sess, batch_size=256)
    a = sess.run(dataset.feed, feed_dict={dataset.train_test_handle: dataset.training_handle})
    b = sess.run(dataset.feed, feed_dict={dataset.train_test_handle: dataset.seen_handle})
    print('hehe')


if __name__ == '__main__':
    test_2()
