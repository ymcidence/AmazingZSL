import numpy as np
import torch as th
from util.data.array_reader import ArrayReader
DEVICE = th.device("cuda" if th.cuda.is_available()  else "cpu")

class ToyReader(ArrayReader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.content = ['feat', 'label', 'label_emb', 's_cls', 'u_cls', 'cls_emb', 's_cls_emb', 'u_cls_emb']
        self.parts = ['training', 'seen', 'unseen']
        assert self.batch_size % 12 == 0
        self.padding = kwargs.get('padding', True)
        self.seen_emb = np.asarray([[0, 1], [0, 0], [1, 0]])
        self.unseen_emb = np.asarray([[1, 1]])
        self.seen_cls_num = self.seen_emb.shape[0]
        self.unseen_cls_num = self.unseen_emb.shape[0]

    def get_batch(self, part='training'):
        return 0

    def _build_data(self):
        return 0

    def get_batch_tensor(self, part='training'):
        seen_centers = np.repeat(self.seen_emb, self.batch_size // self.seen_cls_num, axis=0)
        unseen_centers = np.repeat(self.unseen_emb, self.batch_size // self.unseen_cls_num, axis=0)

        seen_labels = np.repeat(np.asarray([0, 1, 2], dtype=np.float32), self.batch_size // self.seen_cls_num, axis=0)
        unseen_labels = np.repeat(np.asarray([3], dtype=np.float32), self.batch_size // self.unseen_cls_num, axis=0)

        gaussian_sample_1 = np.random.randn(self.batch_size, 2) / 3
        gaussian_sample_2 = np.random.randn(self.batch_size, 2) / 3

        seen_samples = seen_centers * 2 - 1 + gaussian_sample_1
        unseen_samples = unseen_centers * 2 - 1 + gaussian_sample_2

        if self.padding:
            padding = np.zeros([self.batch_size, 2], np.float32)
            seen_samples = np.concatenate([seen_samples, padding], axis=1)
            unseen_samples = np.concatenate([unseen_samples, padding], axis=1)

        if part == 'unseen':
            feed_list = [unseen_samples,
                         unseen_labels,
                         unseen_centers,
                         np.asarray([0, 1, 2]),
                         np.asarray([3]),
                         np.concatenate([self.seen_emb, self.unseen_emb], axis=0),
                         self.seen_emb,
                         self.unseen_emb]
        else:
            feed_list = [seen_samples,
                         seen_labels,
                         seen_centers,
                         np.asarray([0, 1, 2]),
                         np.asarray([3]),
                         np.concatenate([self.seen_emb, self.unseen_emb], axis=0),
                         self.seen_emb,
                         self.unseen_emb]

        feat = [th.tensor(i, dtype=th.float32).to(DEVICE) for i in feed_list]

        return feat


if __name__ == '__main__':
    data = ToyReader(batch_size=128 * 3)
    batch = data.get_batch_tensor('training')
    print('hehe')
