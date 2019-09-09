import tensorflow as tf
import os
from model.basic_model import calibration_loss, partial_semantic_cls
from model.gan_model import SimpleGanModel
from util.data import set_profiles
from util.layer import conventional_layers as layers
from util.eval import zsl_acc
from time import gmtime, strftime


class BidirectionalModel(SimpleGanModel):
    def _get_feat(self):
        # 1. data batch
        self.feat = tf.identity(self.data.feed['feat'])
        self.label_emb = tf.identity(self.data.feed['label_emb'])
        self.label = tf.one_hot(tf.identity(self.data.feed['label']), depth=self.cls_num)
        self.cls_emb = tf.identity(self.data.feed['cls_emb'])
        self.s_cls = tf.identity(self.data.feed['s_cls'])
        self.u_cls = tf.identity(self.data.feed['u_cls'])

        # 2. sizes
        self.batch_size = tf.shape(self.feat)[0]
        self.feat_size = set_profiles.FEAT_DIM[self.set_name]
        self.emb_size = set_profiles.ATTR_DIM[self.set_name]
        comp_size = self.feat_size - self.emb_size

        self.mid_size = self.feat_size // 2

        # 3. random padding
        self.z_random = tf.random_normal([self.batch_size, comp_size], mean=0., stddev=1)
        self.z_padding_1 = tf.random_normal([self.batch_size, self.feat_size - self.mid_size], mean=0., stddev=1)
        self.z_padding_2 = tf.random_normal([self.batch_size, self.mid_size - self.emb_size], mean=0., stddev=1)
        self.zero_padding = tf.zeros([self.cls_num, self.mid_size - self.emb_size], dtype=tf.float32)
        self.random_label = tf.one_hot(
            tf.random.uniform([self.batch_size], minval=0, maxval=self.cls_num - 1, dtype=tf.int32), depth=self.cls_num)
        self.random_emb = tf.stop_gradient(self.random_label @ self.cls_emb)

    # def _forward_path(self, feat, inn, slice_dim):
    #

    def _build_net(self):
        self._get_feat()
        inn_1 = self.InnModule('inn_1', self.feat_size // 2)
        inn_2 = self.InnModule('inn_2', self.feat_size // 4)

        with tf.variable_scope('actor') as scope:
            # 1. v->s
            self.mid_s, _ = inn_1(self.feat, 0)
            self.mid_s_1 = self.mid_s[:, :self.mid_size]
            self.mid_s_2 = self.mid_s[:, self.mid_size:]

            self.pred_s, _ = inn_2(self.mid_s_1, 0)
            self.pred_s_1 = self.pred_s[:, :self.emb_size]
            self.pred_s_2 = self.pred_s[:, self.emb_size:]

            # 2. s->v'
            scope.reuse_variables()
            connected_2 = tf.concat([self.random_emb, self.z_padding_2], 1)
            self.mid_v, _ = inn_2(connected_2, 0, forward=False)

            connected_1 = tf.concat([self.mid_v, self.z_padding_1], 1)
            self.pred_v, _ = inn_1(connected_1, 0, forward=False)

            # 3. v'->s'
            connected_e_1 = tf.concat([self.cls_emb, self.zero_padding], 1)
            self.mid_e, _ = inn_2(connected_e_1, 0, forward=False)

        with tf.variable_scope('critic') as scope:
            # 1. real
            connected_emb = tf.concat([connected_2, self.z_padding_1], axis=1)
            self.d_v_real = tf.sigmoid(layers.fc_layer('fc_v', self.feat, 1))
            self.d_s_real = tf.sigmoid(layers.fc_layer('fc_s', connected_emb, 1))
            scope.reuse_variables()
            # 2. fake
            connected_s = tf.concat([self.pred_s, self.mid_s_2], axis=1)
            self.d_v_fake = tf.sigmoid(layers.fc_layer('fc_v', self.pred_v, 1))
            self.d_s_fake = tf.sigmoid(layers.fc_layer('fc_s', connected_s, 1))

    def _build_loss(self):
        with tf.name_scope('actor'):
            # 1. regression loss
            reg_loss = tf.nn.l2_loss(self.pred_s_1 - self.label_emb)
            # 2. cls loss
            cls_loss = partial_semantic_cls(self.mid_s_1, self.mid_e, self.label, self.s_cls, self.soft_max_temp)
            cal_loss = calibration_loss(self.mid_s_1, self.mid_e, self.u_cls, self.soft_max_temp)
            # 3. gan loss
            z_loss_fake = (tf.reduce_mean(self.d_v_fake) + tf.reduce_mean(self.d_s_fake)) * -1

            loss_mid = tf.reduce_mean(cls_loss) + self.cali * cal_loss

            loss_z = self.lamb * z_loss_fake

            loss_actor = reg_loss + loss_mid + loss_z

            tf.summary.scalar('loss_v', loss_mid)
            tf.summary.scalar('loss_z', loss_z)
            tf.summary.scalar('cal_loss', cal_loss)
            tf.summary.scalar('loss_actor', loss_actor)

        with tf.name_scope('critic'):
            p_loss = tf.reduce_mean(self.d_s_fake) + tf.reduce_mean(self.d_v_fake) - tf.reduce_mean(
                self.d_s_real) - tf.reduce_mean(self.d_v_real)
            loss_critic = self.lamb * p_loss

            tf.summary.scalar('loss_critic', loss_critic)

        return loss_actor, loss_critic

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
        a_summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope='actor'))
        c_summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope='critic'))
        for i in range(max_iter):
            # 1. train
            feed_dict = {self.data.train_test_handle: self.data.training_handle}
            # 1.1 actor
            s_value, label_value, emb_value, a_loss_value, c_loss_value, _, a_summary_value, c_summary_value, step_value = self.sess.run(
                [self.mid_s_1, self.label,
                 self.mid_e, self.actor_loss,
                 self.critic_loss, opt[1],
                 a_summary, c_summary,
                 self.global_step],
                feed_dict=feed_dict)

            writer.add_summary(a_summary_value, step_value // 2)
            writer.add_summary(c_summary_value, step_value // 2)

            if (i + 1) % 50 == 0:
                seen_dict = {self.data.train_test_handle: self.data.seen_handle}
                unseen_dict = {self.data.train_test_handle: self.data.unseen_handle}

                seen_s, seen_label, e_value = self.sess.run([self.mid_s_1, self.label, self.mid_e],
                                                            feed_dict=seen_dict)
                unseen_s, unseen_label = self.sess.run([self.mid_s_1, self.label], feed_dict=unseen_dict)

                train_acc = zsl_acc.cls_wise_acc(s_value, label_value, emb_value)
                seen_acc = zsl_acc.cls_wise_acc(seen_s, seen_label, e_value)
                unseen_acc = zsl_acc.cls_wise_acc(unseen_s, unseen_label, e_value)
                h_score = zsl_acc.h_score(seen_acc, unseen_acc)

                hook_summary = tf.Summary(value=[tf.Summary.Value(tag='hook/train_acc', simple_value=train_acc)])
                writer.add_summary(hook_summary, step_value)
                hook_summary = tf.Summary(value=[tf.Summary.Value(tag='hook/seen_acc', simple_value=seen_acc)])
                writer.add_summary(hook_summary, step_value)
                hook_summary = tf.Summary(value=[tf.Summary.Value(tag='hook/unseen_acc', simple_value=unseen_acc)])
                writer.add_summary(hook_summary, step_value)
                hook_summary = tf.Summary(value=[tf.Summary.Value(tag='hook/h_score', simple_value=h_score)])
                writer.add_summary(hook_summary, step_value)
                print('Step {}, Actor Loss {}, Critic Loss {}'.format(step_value, a_loss_value, c_loss_value))
