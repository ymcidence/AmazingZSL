import tensorflow as tf
import os
from util.data.dataset import Dataset
from util.data import set_profiles
from util.layer import inn
from util.layer import mmd
from util.layer import conventional_layers as layers


def semantic_cls(tensor_in: tf.Tensor, cls_emb: tf.Tensor, label: tf.Tensor, temp=.5):
    distances = mmd.distance(tensor_in, tf.stop_gradient(cls_emb)) / temp
    return tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=distances)


def partial_semantic_cls(tensor_in: tf.Tensor, cls_emb: tf.Tensor, label: tf.Tensor, part_entry, temp=.5):
    """

    :param tensor_in: [N D]
    :param cls_emb: [C M]
    :param label: [N C] one-hot
    :param part_entry: [C_]
    :param temp: temperature
    :return:
    """
    cls_emb = tf.gather(cls_emb, indices=part_entry, axis=0)  # [C_ M]
    label = tf.gather(label, indices=part_entry, axis=1)  # [N C_]

    return semantic_cls(tensor_in, cls_emb, label, temp)


class BasicModel(object):
    def __init__(self, **kwargs):
        self.sess = kwargs.get('sess', tf.Session())
        self.set_name = kwargs.get('set_name', 'AWA1')
        self.gan = kwargs.get('gan', True)
        self.soft_max_temp = kwargs.get('temp', .5)
        self.seen_num = set_profiles.LABEL_NUM[self.set_name][0]
        self.unseen_num = set_profiles.LABEL_NUM[self.set_name][1]
        self.cls_num = self.seen_num + self.unseen_num
        self.batch_size = kwargs.get('batch_size', 256)
        self.data = Dataset(set_name=self.set_name, batch_size=self.batch_size, sess=self.sess)

    def _get_feat(self):
        self.feat = tf.identity(self.data.feed['feat'])
        self.label_emb = tf.identity(self.data.feed['label_emb'])
        self.label = tf.one_hot(tf.identity(self.data.feed['label']), depth=self.cls_num)
        self.cls_emb = tf.identity(self.data.feed['cls_emb'])
        self.s_cls = tf.identity(self.data.feed['s_cls'])
        self.u_cls = tf.identity(self.data.feed['u_cls'])
        self.batch_size = tf.shape(self.feat)[0]
        self.feat_size = set_profiles.FEAT_DIM[self.set_name]
        self.emb_size = set_profiles.ATTR_DIM[self.set_name]
        comp_size = self.feat_size - self.emb_size
        self.z_random = tf.random_normal([self.batch_size, comp_size], mean=0., stddev=.5)
        self.z_padding = tf.random_normal([self.batch_size, comp_size], mean=0., stddev=.5)
        self.random_label = tf.one_hot(
            tf.random.uniform([self.batch_size], minval=0, maxval=self.cls_num - 1, dtype=tf.int32), depth=self.cls_num)
        self.random_emb = tf.stop_gradient(self.random_label @ self.cls_emb)

    def _build_net(self):
        self._get_feat()

        with tf.variable_scope('actor') as scope:
            # 1. v->s
            self.pred_s, self.det_1 = inn.invertible_projection('inv_1', self.feat, 0)
            self.pred_s_1 = self.pred_s[:, :self.emb_size]
            self.pred_s_2 = self.pred_s[:, self.emb_size:]
            # 2. s->v
            scope.reuse_variables()
            connected_emb = tf.concat([self.random_emb, self.z_padding], 1)
            self.pred_v, self.det_2 = inn.invertible_projection('inv_1', connected_emb, 0, forward=False)

        if self.gan:
            with tf.variable_scope('critic') as scope:
                self.d_real = tf.sigmoid(layers.fc_layer('fc_1', self.feat, 1))
                scope.reuse_variables()
                self.d_fake = tf.sigmoid(layers.fc_layer('fc_1', self.pred_v, 1))

    def _build_loss_no_gan(self):
        with tf.name_scope('actor'):
            cls_loss = partial_semantic_cls(self.pred_s_1, self.cls_emb, self.label, self.s_cls, self.soft_max_temp)
            mmd_loss_z = mmd

        loss = cls_loss

        return loss

    def _build_loss_gan(self):
        pass

    def train(self):
        pass

    def save(self, task_name, step, var_list=None):
        save_path = os.path.join('./result', self.set_name, 'model')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_name = os.path.join(save_path, task_name, 'ymmodel')
        saver = tf.train.Saver(var_list=var_list)
        saver.save(self.sess, save_name, step)

    def restore(self, save_path, var_list=None):
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(self.sess, save_path=save_path)
