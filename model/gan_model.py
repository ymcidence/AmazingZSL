import tensorflow as tf
import os
from time import gmtime, strftime
from util.layer import conventional_layers as layers
from util.eval import zsl_acc
from model.basic_model import BasicModel, calibration_loss, partial_semantic_cls


class SimpleGanModel(BasicModel):
    def _build_net(self):
        self._get_feat()
        inn = self.InnModule('inn', int(self.feat_size / 2), depth=self.depth, coupling=self.coupling,
                             permute=self.permute, norm=self.norm)

        with tf.variable_scope('actor') as scope:
            # 1. v->s'
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
            self.pred_ss, _ = inn(self.pred_v, 0)
            self.pred_ss_1 = self.pred_ss[:, :self.emb_size]

        with tf.variable_scope('critic') as scope:
            self.d_v_real = tf.sigmoid(layers.fc_layer('fc_v', self.feat, 1))
            self.d_s_real = tf.sigmoid(layers.fc_layer('fc_s', connected_emb, 1))
            scope.reuse_variables()
            self.d_v_fake = tf.sigmoid(layers.fc_layer('fc_v', self.pred_v, 1))
            self.d_s_fake = tf.sigmoid(layers.fc_layer('fc_s', self.pred_s, 1))

    def _build_loss(self):
        with tf.name_scope('actor'):
            cls_loss = partial_semantic_cls(self.pred_s_1, self.cls_emb, self.label, self.s_cls, self.soft_max_temp)
            cal_loss = calibration_loss(self.pred_s_1, self.cls_emb, self.u_cls, self.soft_max_temp)

            z_loss_fake = (tf.reduce_mean(self.d_v_fake) + tf.reduce_mean(self.d_s_fake)) * -1

            loss_v = tf.reduce_mean(cls_loss) + self.cali * cal_loss

            loss_z = self.lamb * z_loss_fake  # - self.det_1

            loss_actor = loss_v + loss_z

            tf.summary.scalar('loss_v', loss_v)
            tf.summary.scalar('loss_z', loss_z)
            tf.summary.scalar('cal_loss', cal_loss)
            tf.summary.scalar('loss_actor', loss_actor)

        with tf.name_scope('critic'):
            p_loss = tf.reduce_mean(self.d_s_fake) + tf.reduce_mean(self.d_v_fake) - tf.reduce_mean(
                self.d_s_real) - tf.reduce_mean(self.d_v_real)
            loss_critic = self.lamb * p_loss

            tf.summary.scalar('loss_critic', loss_critic)

        return loss_actor, loss_critic

    def _build_opt(self):
        self.actor_loss, self.critic_loss = self._build_loss()
        actor_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        critic_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        actor_opt = tf.train.RMSPropOptimizer(self.lr).minimize(self.actor_loss, self.global_step, var_list=actor_var)
        with tf.control_dependencies([actor_opt]):
            critic_opt_0 = tf.train.RMSPropOptimizer(self.lr).minimize(self.critic_loss, self.global_step,
                                                                       var_list=critic_var)

        with tf.control_dependencies([critic_opt_0]):
            critic_clipping = [tf.assign(var, tf.clip_by_value(var, -0.01, 0.01)) for var in critic_var]
            critic_opt = tf.tuple(critic_clipping)

        return actor_opt, critic_opt

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
            s_f = self.pred_s_1 if self.cls_from == 's' else self.feat
            s_e = self.cls_emb if self.cls_from == 's' else self.pred_ve

            # 1.1 actor
            s_value, label_value, emb_value, a_loss_value, c_loss_value, _, a_summary_value, c_summary_value, step_value = self.sess.run(
                [s_f, self.label,
                 s_e, self.actor_loss,
                 self.critic_loss, opt[1],
                 a_summary, c_summary,
                 self.global_step],
                feed_dict=feed_dict)

            writer.add_summary(a_summary_value, step_value // 2)
            writer.add_summary(c_summary_value, step_value // 2)

            if (i + 1) % 50 == 0:
                seen_dict = {self.data.train_test_handle: self.data.seen_handle}
                unseen_dict = {self.data.train_test_handle: self.data.unseen_handle}

                seen_s, seen_label, e_value = self.sess.run([s_f, self.label, s_e], feed_dict=seen_dict)
                unseen_s, unseen_label = self.sess.run([s_f, self.label], feed_dict=unseen_dict)

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


class ImprovedGanModel(SimpleGanModel):
    def __init__(self, **kwargs):
        self.lamb2 = kwargs.get('lamb2', 10)
        super().__init__(**kwargs)

    def _build_net(self):
        self._get_feat()
        inn = self.InnModule('inn', int(self.feat_size / 2), depth=self.depth, coupling=self.coupling,
                             permute=self.permute, norm=self.norm)
        eps = tf.random_uniform([], 0.0, 1.0)

        with tf.variable_scope('actor') as scope:
            # 1. v->s
            self.pred_s, self.det_1 = inn(self.feat, 0)
            self.pred_s_1 = self.pred_s[:, :self.emb_size]
            self.pred_s_2 = self.pred_s[:, self.emb_size:]
            # 2. s->v'
            scope.reuse_variables()
            connected_emb = tf.concat([self.random_emb, self.z_padding], 1)
            self.pred_v, self.det_2 = inn(connected_emb, 0, forward=False)
            # 3. v'->s'
            self.pred_ss, _ = inn(tf.stop_gradient(self.pred_v), 0)
            self.pred_ss_1 = self.pred_ss[:, :self.emb_size]

        with tf.variable_scope('critic') as scope:
            # 1. real
            self.d_v_real = tf.sigmoid(layers.fc_layer('fc_v', self.feat, 1))
            self.d_s_real = tf.sigmoid(layers.fc_layer('fc_s', connected_emb, 1))
            scope.reuse_variables()
            # 2. fake
            self.d_v_fake = tf.sigmoid(layers.fc_layer('fc_v', self.pred_v, 1))
            self.d_s_fake = tf.sigmoid(layers.fc_layer('fc_s', self.pred_s, 1))

            # 3. improved wgan component
            self.v_hat = self.feat * eps + (1. - eps) * self.pred_v
            self.s_hat = connected_emb * eps + (1. - eps) * self.pred_s
            self.d_v_hat = tf.sigmoid(layers.fc_layer('fc_v', self.v_hat, 1))
            self.d_s_hat = tf.sigmoid(layers.fc_layer('fc_s', self.s_hat, 1))

    def _build_loss(self):
        with tf.name_scope('actor'):
            cls_loss = partial_semantic_cls(self.pred_s_1, self.cls_emb, self.label, self.s_cls, self.soft_max_temp)
            cal_loss = calibration_loss(self.pred_s_1, self.cls_emb, self.u_cls, self.soft_max_temp)

            z_loss_fake = (tf.reduce_mean(self.d_v_fake) + tf.reduce_mean(self.d_s_fake)) * -1

            loss_v = tf.reduce_mean(cls_loss) + self.cali * cal_loss

            loss_z = z_loss_fake * self.lamb  # - self.det_1

            loss_actor = loss_v + loss_z

            tf.summary.scalar('loss_v', loss_v)
            tf.summary.scalar('loss_z', loss_z)
            tf.summary.scalar('cal_loss', cal_loss)
            tf.summary.scalar('loss_actor', loss_actor)

        with tf.name_scope('critic'):
            p_loss = tf.reduce_mean(self.d_s_fake) + tf.reduce_mean(self.d_v_fake) - tf.reduce_mean(
                self.d_s_real) - tf.reduce_mean(self.d_v_real)
            loss_critic = p_loss * self.lamb

            v_hat_g = tf.gradients(self.d_v_hat, self.v_hat)[0]
            s_hat_g = tf.gradients(self.d_s_hat, self.s_hat)[0]

            v_wgan_loss = tf.sqrt(tf.reduce_sum(tf.square(v_hat_g), reduction_indices=[1]) + 1e-8)
            v_wgan_loss = tf.reduce_mean((v_wgan_loss - 1.0) ** 2)

            s_wgan_loss = tf.sqrt(tf.reduce_sum(tf.square(s_hat_g), reduction_indices=[1]) + 1e-8)
            s_wgan_loss = tf.reduce_mean((s_wgan_loss - 1.0) ** 2)

            wgan_loss = (s_wgan_loss + v_wgan_loss) * self.lamb2

            tf.summary.scalar('loss_critic', loss_critic)
            tf.summary.scalar('wgan_loss', wgan_loss)

            loss_critic = loss_critic + wgan_loss

        return loss_actor, loss_critic

    def _build_opt(self):
        self.actor_loss, self.critic_loss = self._build_loss()
        actor_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        critic_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        actor_opt = tf.train.RMSPropOptimizer(2e-4).minimize(self.actor_loss, self.global_step, var_list=actor_var)
        with tf.control_dependencies([actor_opt]):
            critic_opt = tf.train.RMSPropOptimizer(2e-4).minimize(self.critic_loss, self.global_step,
                                                                  var_list=critic_var)

        return actor_opt, critic_opt


class ComplexGanModel(SimpleGanModel):
    def __build_net(self):
        self._get_feat()
        inn = self.InnModule('inn', int(self.feat_size / 2))

        with tf.variable_scope('actor') as scope:
            # 1. v->s'
            self.pred_s, self.det_1 = inn(self.feat, 0)
            self.pred_s_1 = self.pred_s[:, :self.emb_size]
            self.pred_s_2 = self.pred_s[:, self.emb_size:]
            # 2. s->v'
            scope.reuse_variables()
            connected_emb = tf.concat([self.random_emb, self.z_padding], 1)
            self.pred_v, self.det_2 = inn(connected_emb, 0, forward=False)
            # 3. s'->v'
            connected_emb_2 = tf.concat([self.pred_s_1, self.z_padding], 1)
            self.pred_vv, _ = inn(tf.stop_gradient(connected_emb_2), 0)
            self.pred_vv_1 = self.pred_vv[:, :self.emb_size]

        with tf.variable_scope('critic') as scope:
            self.d_v_real = tf.sigmoid(layers.fc_layer('fc_v', self.feat, 1))
            self.d_s_real = tf.sigmoid(layers.fc_layer('fc_s', connected_emb, 1))
            scope.reuse_variables()
            v_fake = tf.concat([self.pred_v, self.pred_vv], 0)
            self.d_v_fake = tf.sigmoid(layers.fc_layer('fc_v', v_fake, 1))
            self.d_s_fake = tf.sigmoid(layers.fc_layer('fc_s', self.pred_s, 1))

    def _build_loss(self):
        with tf.name_scope('actor'):
            cls_loss = partial_semantic_cls(self.pred_s_1, self.cls_emb, self.label, self.s_cls, self.soft_max_temp)
            cal_loss = calibration_loss(self.pred_s_1, self.cls_emb, self.u_cls, self.soft_max_temp)

            unseen_vv = layers.label_select(self.pred_ss_1, self.random_label, self.u_cls, self.cls_num)
            cal_u_loss = calibration_loss(unseen_vv, self.cls_emb, self.u_cls, self.soft_max_temp)

            z_loss_fake = (tf.reduce_mean(self.d_v_fake) + tf.reduce_mean(self.d_s_fake)) * -1

            loss_v = tf.reduce_mean(cls_loss) + self.cali * cal_loss - 0.1 * cal_u_loss

            loss_z = self.lamb * z_loss_fake  # - self.det_1

            loss_actor = loss_v + loss_z

            tf.summary.scalar('loss_v', loss_v)
            tf.summary.scalar('loss_z', loss_z)
            tf.summary.scalar('cal_loss', cal_loss)
            tf.summary.scalar('cal_u_loss', cal_u_loss)
            tf.summary.scalar('loss_actor', loss_actor)

        with tf.name_scope('critic'):
            p_loss = tf.reduce_mean(self.d_s_fake) + tf.reduce_mean(self.d_v_fake) - tf.reduce_mean(
                self.d_s_real) - tf.reduce_mean(self.d_v_real)

            loss_critic = self.lamb * p_loss

            tf.summary.scalar('loss_critic', loss_critic)

        return loss_actor, loss_critic
