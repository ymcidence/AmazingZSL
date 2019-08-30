import tensorflow as tf
import os
from util.data.dataset import Dataset


class BasicModel(object):
    def __init__(self, **kwargs):
        self.sess = kwargs.get('sess', tf.Session())
        self.set_name = kwargs.get('set_name', 'AWA1')
        self.batch_size = kwargs.get('batch_size', 256)
        self.data = Dataset(set_name=self.set_name, batch_size=self.batch_size, sess=self.sess)

    def _build_net(self):
        pass

    def _build_loss(self):
        pass

    def train(self):
        pass

    def save(self, task_name, step, var_list=None):
        save_name = os.path.join('./result', self.set_name, 'model', task_name, 'ymmodel')
        saver = tf.train.Saver(var_list=var_list)
        saver.save(self.sess, save_name, step)
