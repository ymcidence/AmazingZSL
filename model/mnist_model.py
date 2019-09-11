import tensorflow as tf
from util.flow.inn_module import SimpleINN
from util.data.mnist import Dataset
from util.layer.mmd import basic_mmd
from time import gmtime, strftime
import os


class MNISTModel(object):
    def __init__(self, batch_size=256, sess=tf.Session()):
        self.dim = int(28 * 28)
        self.cls_num = 10
        self.sess = sess
        self.batch_size = batch_size
        self.data = Dataset(sess=sess, batch_size=batch_size)
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
        self.random_label = tf.one_hot(
            tf.random.uniform([self.batch_size], minval=0, maxval=self.cls_num - 1, dtype=tf.int32), depth=self.cls_num)
        self.random_condition = tf.concat([self.random_label, self.z_padding], axis=1)

    def _build_net(self):
        self._get_feat()
        self.inn = SimpleINN('mnist', self.dim, depth=1)
        with tf.variable_scope('mnist') as scope:
            self.yz_hat, _ = self.inn(self.feat, 0)
            self.y_hat = self.yz_hat[:, :self.emb_size]
            self.z_hat = self.yz_hat[:, self.emb_size:]

            scope.reuse_variables()
            self.x_hat, _ = self.inn(tf.stop_gradient(self.yz_hat), 0, False)
            self.gen, _ = self.inn(self.random_condition, 0, False)

        gen_image = tf.nn.relu(tf.reshape(self.gen, [-1, 28, 28, 1]))
        recon_image = tf.nn.relu(tf.reshape(self.x_hat, [-1, 28, 28, 1]))
        origin_image = tf.reshape(self.feat, [-1, 28, 28, 1])

        tf.summary.image('img', origin_image, max_outputs=1)
        tf.summary.image('recon', recon_image, max_outputs=1)
        tf.summary.image('gen', gen_image, max_outputs=1)

    def _build_loss(self):
        with tf.name_scope('actor'):
            cls_loss = tf.nn.l2_loss(self.y_hat - self.label)
            z_loss = basic_mmd(self.yz_hat, self.random_condition, scale=0.025)
            x_loss = basic_mmd(self.gen, self.feat, scale=0.025) + basic_mmd(self.x_hat, self.feat, scale=0.025)
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

    def _build_opt(self):
        self.loss = self._build_loss()
        adam = tf.train.AdamOptimizer()
        opt = adam.minimize(self.loss, self.global_step)
        return opt

    def train(self, restore_file=None, restore_list=None, task='hehe1', max_iter=500000):
        opt = self._build_opt()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
        summary_path = os.path.join('./result/mnist', 'log', task + '_' + time_string) + os.sep
        save_path = os.path.join('./result/mnist', task + '_' + 'model') + os.sep

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


if __name__ == '__main__':
    model = MNISTModel()
    model.train(task='reg_v')
