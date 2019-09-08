import tensorflow as tf
import os
from util.data.dataset import Dataset
from util.data import set_profiles
from util.flow import inn_module
from util.layer import mmd
from util.layer import conventional_layers as layers
from util.eval import zsl_acc
from time import gmtime, strftime


def semantic_cls(tensor_in: tf.Tensor, cls_emb: tf.Tensor, label: tf.Tensor, temp=.5):
    distances = -1 * mmd.distance(tensor_in, tf.stop_gradient(cls_emb)) / temp
    return tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=distances)


def partial_semantic_cls(tensor_in: tf.Tensor, cls_emb: tf.Tensor, label: tf.Tensor, part_entry, temp=.5):
    """

    :param tensor_in: [N D]
    :param cls_emb: [C M]
    :param label: [N C] one-hot
    :param part_entry: [C_s]
    :param temp: temperature
    :return:
    """
    cls_emb = tf.gather(cls_emb, indices=part_entry, axis=0)  # [C_s M]
    label = tf.gather(label, indices=part_entry, axis=1)  # [N C_s]

    return semantic_cls(tensor_in, cls_emb, label, temp)


def calibration_loss(tensor_in: tf.Tensor, cls_emb: tf.Tensor, part_entry, temp=.5):
    unseen_cls_emb = tf.gather(cls_emb, indices=part_entry, axis=0)  # [C_u M]
    distances = -1 * mmd.distance(tensor_in, unseen_cls_emb) / temp
    prob = tf.nn.softmax(distances + 1e-8)
    ent = - prob * tf.log(prob + 1e-8)
    return tf.reduce_mean(ent)


class BasicModel(object):
    def __init__(self, **kwargs):
        self.sess = kwargs.get('sess', tf.Session())
        self.set_name = kwargs.get('set_name', 'AWA1')
        self.gan = kwargs.get('gan', True)
        self.soft_max_temp = kwargs.get('temp', .5)
        self.lamb = kwargs.get('lamb', 3)
        self.cls_from = kwargs.get('cls_from', 's')
        self.seen_num = set_profiles.LABEL_NUM[self.set_name][0]
        self.unseen_num = set_profiles.LABEL_NUM[self.set_name][1]
        self.cls_num = self.seen_num + self.unseen_num
        self.batch_size = kwargs.get('batch_size', 256)
        self.InnModule = kwargs.get('inn_module', inn_module.SimpleINN)
        self.data = Dataset(set_name=self.set_name, batch_size=self.batch_size, sess=self.sess)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self._build_net()

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
        self.zero_padding = tf.zeros([self.cls_num, comp_size], dtype=tf.float32)
        self.random_label = tf.one_hot(
            tf.random.uniform([self.batch_size], minval=0, maxval=self.cls_num - 1, dtype=tf.int32), depth=self.cls_num)
        self.random_emb = tf.stop_gradient(self.random_label @ self.cls_emb)

    def _build_net(self):
        self._get_feat()
        inn = self.InnModule('inn', int(self.feat_size / 2))

        with tf.variable_scope('actor') as scope:
            # 1. v->s
            self.pred_s, self.det_1 = inn(self.feat, 0)
            self.pred_s_1 = self.pred_s[:, :self.emb_size]
            self.pred_s_2 = self.pred_s[:, self.emb_size:]
            # 2. s->v'
            scope.reuse_variables()
            connected_emb = tf.concat([self.random_emb, self.z_padding], 1)
            self.pred_v, self.det_2 = inn(connected_emb, 0, forward=False)
            ve = tf.concat([self.cls_emb, self.zero_padding], 1)
            self.pred_ve, _ = inn(ve, 0, forward=False)
            # 3. v'->s'
            self.pred_ss, _ = inn(tf.stop_gradient(self.pred_v), 0)
            self.pred_ss_1 = self.pred_ss[:, :self.emb_size]

        if self.gan:
            with tf.variable_scope('critic') as scope:
                self.d_real = tf.sigmoid(layers.fc_layer('fc_1', self.feat, 1))
                scope.reuse_variables()
                self.d_fake = tf.sigmoid(layers.fc_layer('fc_1', self.pred_v, 1))

    def _build_loss(self):
        with tf.name_scope('forward'):
            cls_loss = partial_semantic_cls(self.pred_s_1, self.cls_emb, self.label, self.s_cls, self.soft_max_temp)
            cal_loss = calibration_loss(self.pred_s_1, self.cls_emb, self.u_cls, self.soft_max_temp)
            # mmd_loss_z = mmd.basic_mmd(self.pred_s, tf.concat([self.label_emb, self.z_random], axis=1), scale=0.025)
            mmd_loss_z = mmd.basic_mmd(self.pred_s_2, self.z_random, scale=0.025)

            loss_v = tf.reduce_mean(cls_loss) + .5 * cal_loss

            loss_z = self.lamb * mmd_loss_z  # - self.det_1

            tf.summary.scalar('loss_v', loss_v)
            tf.summary.scalar('loss_z', loss_z)
            tf.summary.scalar('mmd_loss_z', mmd_loss_z)
            tf.summary.scalar('cal_loss', cal_loss)

        with tf.name_scope('reverse'):
            mmd_loss_x = mmd.basic_mmd(self.pred_v, self.feat)

            loss_px = self.lamb * mmd_loss_x  # - self.det_2

            # tf.summary.scalar('mmd_loss_x', mmd_loss_x)
            # tf.summary.scalar('det_2', self.det_2)
            cls_loss_2 = tf.reduce_mean(
                semantic_cls(self.pred_ss_1, self.cls_emb, self.random_label, self.soft_max_temp))
            loss_x = loss_px + cls_loss_2
            tf.summary.scalar('loss_px', loss_px)
            tf.summary.scalar('cls_loss_2', cls_loss_2)

        loss = loss_v + loss_z + loss_x

        return loss

    def _build_opt(self):
        self.loss = self._build_loss()
        adam = tf.train.RMSPropOptimizer(1e-3)
        return adam.minimize(self.loss, self.global_step)

    def train(self, restore_file=None, restore_list=None, task='hehe1', max_iter=500000):
        opt = self._build_opt()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
        summary_path = os.path.join('./result', self.set_name, 'log', task + '_' + time_string) + os.sep
        save_path = os.path.join('./result', self.set_name, task + '_' + 'model') + os.sep

        if restore_file is not None:
            self._restore(restore_file, restore_list)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        writer = tf.summary.FileWriter(summary_path, graph=self.sess.graph)
        summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))
        for i in range(max_iter):
            feed_dict = {self.data.train_test_handle: self.data.training_handle}

            s_f = self.pred_s_1 if self.cls_from == 's' else self.feat
            s_e = self.cls_emb if self.cls_from == 's' else self.pred_ve

            s_value, label_value, emb_value, loss_value, _, summary_value, step_value = self.sess.run(
                [s_f, self.label,
                 s_e, self.loss, opt, summary,
                 self.global_step],
                feed_dict=feed_dict)
            writer.add_summary(summary_value, step_value)
            if (i + 1) % 50 == 0:
                seen_dict = {self.data.train_test_handle: self.data.seen_handle}
                unseen_dict = {self.data.train_test_handle: self.data.unseen_handle}

                seen_s, seen_label = self.sess.run([s_f, self.label], feed_dict=seen_dict)
                unseen_s, unseen_label = self.sess.run([s_f, self.label], feed_dict=unseen_dict)

                train_acc = zsl_acc.cls_wise_acc(s_value, label_value, emb_value)
                seen_acc = zsl_acc.cls_wise_acc(seen_s, seen_label, emb_value)
                unseen_acc = zsl_acc.cls_wise_acc(unseen_s, unseen_label, emb_value)
                h_score = zsl_acc.h_score(seen_acc, unseen_acc)

                hook_summary = tf.Summary(value=[tf.Summary.Value(tag='hook/train_acc', simple_value=train_acc)])
                writer.add_summary(hook_summary, step_value)
                hook_summary = tf.Summary(value=[tf.Summary.Value(tag='hook/seen_acc', simple_value=seen_acc)])
                writer.add_summary(hook_summary, step_value)
                hook_summary = tf.Summary(value=[tf.Summary.Value(tag='hook/unseen_acc', simple_value=unseen_acc)])
                writer.add_summary(hook_summary, step_value)
                hook_summary = tf.Summary(value=[tf.Summary.Value(tag='hook/h_score', simple_value=h_score)])
                writer.add_summary(hook_summary, step_value)
                print('Step {}, Loss {}'.format(step_value, loss_value))

    def save(self, task_name, step, var_list=None):
        save_path = os.path.join('./result', self.set_name, 'model')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_name = os.path.join(save_path, task_name, 'ymmodel')
        saver = tf.train.Saver(var_list=var_list)
        saver.save(self.sess, save_name, step)

    def _restore(self, save_path, var_list=None):
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(self.sess, save_path=save_path)
