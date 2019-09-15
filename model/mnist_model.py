import tensorflow as tf
# noinspection PyUnresolvedReferences
from util.flow.inn_module import SimpleINN, SimplerINN
from util.data.mnist import Dataset
from util.layer import mmd
from time import gmtime, strftime
import os
from util.layer import conventional_layers as layers


class MNISTModel(object):
    def __init__(self, batch_size=256, sess=tf.Session()):
        self.dim = int(28 * 28)
        self.cls_num = 10
        self.sess = sess
        self.batch_size = batch_size
        self.data = Dataset(sess=sess, batch_size=batch_size)
        self.lr = 1e-4
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self._build_net()

    def _get_feat(self):
        self.feat = tf.identity(self.data.feed[0])
        self.label = tf.identity(self.data.feed[1])
        self.feat_size = 784
        self.emb_size = 10
        comp_size = self.feat_size - self.emb_size
        self.z_random = tf.random_normal([self.batch_size, comp_size], mean=0., stddev=1)
        self.z_padding = tf.random_normal([self.batch_size, comp_size], mean=0., stddev=1)
        self.z_z = tf.random_normal([self.batch_size, self.feat_size], mean=0., stddev=1)
        self.random_label = tf.one_hot(
            tf.random.uniform([self.batch_size], minval=0, maxval=self.cls_num - 1, dtype=tf.int32), depth=self.cls_num)
        self.random_condition = tf.concat([self.random_label, self.z_padding], axis=1)

    def _build_net(self):
        self._get_feat()
        self.inn = SimpleINN('mnist', self.dim, depth=2, norm=False, coupling=0, permute=2)
        with tf.variable_scope('mnist') as scope:
            self.yz_hat, self.det = self.inn(self.feat, 0)
            self.y_hat = self.yz_hat[:, :self.emb_size]
            self.z_hat = self.yz_hat[:, self.emb_size:]

            scope.reuse_variables()
            self.x_hat, _ = self.inn(tf.stop_gradient(self.yz_hat), 0, False)
            self.gen, _ = self.inn(self.random_condition, 0, False)
            self.gen_2 = self.inn(self.z_z, 0, False)

        gen_image = tf.nn.relu(tf.reshape(self.gen, [-1, 28, 28, 1]))
        recon_image = tf.nn.relu(tf.reshape(self.x_hat, [-1, 28, 28, 1]))
        origin_image = tf.reshape(self.feat, [-1, 28, 28, 1])

        tf.summary.image('img', origin_image, max_outputs=1)
        tf.summary.image('recon', recon_image, max_outputs=1)
        tf.summary.image('gen', gen_image, max_outputs=1)

    def _build_loss(self):
        with tf.name_scope('actor'):
            # cls_loss = tf.nn.l2_loss(self.y_hat - self.label)
            cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.y_hat))
            z_loss = mmd.basic_mmd(self.z_hat, self.z_random, scale=0.025)
            # x_loss = mmd.basic_mmd(self.gen, self.feat, scale=0.025) + mmd.basic_mmd(self.x_hat, self.feat, scale=0.025)

            x_loss = mmd.category_mmd(self.gen, self.feat, self.random_label, self.label, 'IMQ')
            # x_loss += mmd.basic_mmd(self.x_hat, self.feat, scale=0.025)
            # x_loss += tf.nn.l2_loss(self.x_hat - self.feat) / self.batch_size

            loss = cls_loss + 10 * (z_loss + x_loss)

            tf.summary.scalar('y_loss', cls_loss)
            tf.summary.scalar('z_loss', z_loss)
            tf.summary.scalar('x_loss', x_loss)
            tf.summary.scalar('loss', loss)

            gt = tf.argmax(self.label, 1)
            pred = tf.argmax(self.y_hat, 1)
            acc = tf.reduce_mean(tf.cast(tf.equal(gt, pred), dtype=tf.float32))
            tf.summary.scalar('acc', acc)

        return loss

    def _build_loss_2(self):
        with tf.name_scope('actor'):
            loss_1 = tf.nn.l2_loss(self.yz_hat)
            loss_2 = - tf.reduce_mean(self.det)
            tf.summary.scalar('loss_1', loss_1)
            tf.summary.scalar('loss_2', loss_2)
        return loss_1 + loss_2

    def _build_opt(self):
        self.loss = self._build_loss_2()
        adam = tf.train.AdamOptimizer(5e-4)
        opt = adam.minimize(self.loss, self.global_step)
        return opt

    def train(self, restore_file=None, restore_list=None, task='hehe1', max_iter=500000):
        opt = self._build_opt()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
        summary_path = os.path.join('./result/mnist2', 'log', task + '_' + time_string) + os.sep
        save_path = os.path.join('./result/mnist2', task + '_' + 'model') + os.sep

        if restore_file is not None:
            self._restore(restore_file, restore_list)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        writer = tf.summary.FileWriter(summary_path, graph=self.sess.graph)
        summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))
        for i in range(max_iter):
            feed_dict = {self.data.train_test_handle: self.data.training_handle}

            v = self.sess.run([self.loss, opt, summary, self.global_step], feed_dict=feed_dict)
            if (i + 1) % 10 == 0:
                writer.add_summary(v[-2], v[-1])
                print('Step {} Loss {}'.format(i, v[0]))

    def _restore(self, restore_file, restore_list):
        pass


class MNISTAdvModel(MNISTModel):
    def _build_net(self):
        self._get_feat()
        self.inn = SimplerINN('mnist', self.dim, depth=1, norm=False, coupling=0, permute=1)
        with tf.variable_scope('actor') as scope:
            self.yz_hat, self.det = self.inn(self.feat, 0)
            self.y_hat = self.yz_hat[:, :self.emb_size]
            self.z_hat = self.yz_hat[:, self.emb_size:]

            scope.reuse_variables()
            self.x_hat, _ = self.inn(tf.stop_gradient(self.yz_hat), 0, False)
            self.gen, _ = self.inn(self.random_condition, 0, False)

        with tf.variable_scope('critic')as scope:
            self.d_y_real = tf.sigmoid(layers.fc_layer('fc_y', self.random_condition, 1))

            x_real = tf.concat([self.feat, self.label], 1)
            self.d_x_real = tf.sigmoid(layers.fc_layer('fc_x', x_real, 1))
            scope.reuse_variables()
            self.d_y_fake = tf.sigmoid(layers.fc_layer('fc_y', self.yz_hat, 1))
            x_fake_1 = tf.concat([self.x_hat, self.label], 1)
            x_fake_2 = tf.concat([self.gen, self.random_label], 1)

            x_hat = x_fake_2  # tf.concat([x_fake_1, x_fake_2], 0)
            self.d_x_fake = tf.sigmoid(layers.fc_layer('fc_x', x_hat, 1))

        gen_image = tf.nn.relu(tf.reshape(self.gen, [-1, 28, 28, 1]))
        recon_image = tf.nn.relu(tf.reshape(self.x_hat, [-1, 28, 28, 1]))
        origin_image = tf.reshape(self.feat, [-1, 28, 28, 1])

        tf.summary.image('img', origin_image, max_outputs=1)
        tf.summary.image('recon', recon_image, max_outputs=1)
        tf.summary.image('gen', gen_image, max_outputs=1)
        tf.summary.histogram('y_hat',self.y_hat)
        tf.summary.histogram('y_real', self.label)

        tf.summary.histogram('z_hat', self.z_hat)
        tf.summary.histogram('z_real', self.z_padding)

    def _build_loss(self):
        with tf.name_scope('actor'):
            cls_loss = tf.nn.l2_loss(self.y_hat - self.label)
            # cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.y_hat))
            z_loss_fake = (tf.reduce_mean(self.d_x_fake) + tf.reduce_mean(self.d_y_fake)) * -1

            loss_g = cls_loss + 10 * z_loss_fake

            tf.summary.scalar('y_loss', cls_loss)
            tf.summary.scalar('z_loss', z_loss_fake)
            # tf.summary.scalar('x_loss', x_loss)
            # tf.summary.scalar('loss', loss)

            gt = tf.argmax(self.label, 1)
            pred = tf.argmax(self.y_hat, 1)
            acc = tf.reduce_mean(tf.cast(tf.equal(gt, pred), dtype=tf.float32))
            tf.summary.scalar('acc', acc)

        with tf.name_scope('critic'):
            p_loss = tf.reduce_mean(self.d_x_fake) + tf.reduce_mean(self.d_y_fake) - tf.reduce_mean(
                self.d_x_real) - tf.reduce_mean(self.d_y_real)

            loss_d = 10 * p_loss
        tf.summary.scalar('loss_critic', loss_d)
        return loss_g, loss_d

    def _build_opt(self):
        self.actor_loss, self.critic_loss = self._build_loss()
        actor_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        critic_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        actor_opt = tf.train.RMSPropOptimizer(self.lr).minimize(self.actor_loss, self.global_step, var_list=actor_var)
        with tf.control_dependencies([actor_opt]):
            critic_opt_0 = tf.train.RMSPropOptimizer(self.lr).minimize(self.critic_loss, self.global_step,
                                                                       var_list=critic_var)

        # with tf.control_dependencies([critic_opt_0]):
        #     critic_clipping = [tf.assign(var, tf.clip_by_value(var, -0.01, 0.01)) for var in critic_var]
        #     critic_opt = tf.tuple(critic_clipping)

        return actor_opt, critic_opt_0

    def train(self, restore_file=None, restore_list=None, task='hehe1', max_iter=50000):
        _, opt = self._build_opt()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
        summary_path = os.path.join('./result/mnist2', 'log', task + '_' + time_string) + os.sep
        save_path = os.path.join('./result/mnist2', task + '_' + 'model') + os.sep

        if restore_file is not None:
            self._restore(restore_file, restore_list)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        writer = tf.summary.FileWriter(summary_path, graph=self.sess.graph)
        summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))
        for i in range(max_iter):
            feed_dict = {self.data.train_test_handle: self.data.training_handle}

            v = self.sess.run([self.actor_loss, self.critic_loss, opt, summary, self.global_step], feed_dict=feed_dict)
            if (i + 1) % 10 == 0:
                writer.add_summary(v[-2], v[-1])
                print('Step {} actor {} critic {}'.format(i, v[0], v[1]))


if __name__ == '__main__':
    model = MNISTModel()
    model.train(task='no_con12')
