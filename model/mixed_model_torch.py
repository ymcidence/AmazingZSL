# import FrEIA.framework as ff
# import FrEIA.modules as fm
# import torch.nn as nn
import torch as th
import numpy as np
import os
import general
from torch.utils.tensorboard import SummaryWriter
# from util.data import set_profiles
# from util.data.dataset import ZSLArrayReader as Reader
from util.layer.mmd import mmd_matrix_multiscale
from util.eval import zsl_acc
from util.layer.torch_layers import batch_adjacency, batch_distance
from time import gmtime, strftime
from model.basic_model_torch import BasicTrainable, BasicModule


class MixedTrainable(BasicTrainable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # noinspection PyUnresolvedReferences
        self.inn_2 = BasicModule(self.lr, feat_length=self.feat_size, depth=self.depth).cuda()
        self.trainable = list(self.inn.parameters()) + list(self.inn_2.parameters())
        self.opt = th.optim.Adam(self.trainable, lr=self.lr, weight_decay=1e-5)

    def _step(self, writer: SummaryWriter, step):
        batch_data = self.reader.get_batch_tensor(self.reader.parts[0])
        feat = batch_data[0]
        # label = batch_data[1]
        label_emb = batch_data[2]
        # s_cls = batch_data[3]
        u_cls = batch_data[4]
        cls_emb = batch_data[5]
        noise = th.rand_like(feat).cuda() * 0.008

        adj = batch_adjacency(label_emb)

        # 1. forward
        v_middle, det_middle = self.inn(x=feat + noise)
        dist = batch_distance(v_middle)

        yz_hat, det_yz = self.inn_2(x=v_middle)
        y_hat = yz_hat[:, :self.emb_size]
        z_hat = yz_hat[:, self.emb_size:]

        cls_loss = th.mean((y_hat - label_emb) ** 2)
        jac_loss = -1 * (th.mean(det_yz) + th.mean(det_middle)) / self.feat_size
        z_loss = th.mean(z_hat ** 2) / 2
        middle_loss = th.mean(adj * dist)

        forward_loss = 10 * cls_loss + jac_loss + z_loss + middle_loss

        # 2. reverse
        ud = self.reader.get_batch_tensor(self.reader.parts[2])
        uf = ud[0]
        rand_ul_ind = np.random.randint(0, self.unseen_num, self.batch_size, dtype=np.int32)
        rand_ul = u_cls[rand_ul_ind]
        rand_y = cls_emb[rand_ul.long(), :]
        rand_z = th.randn_like(z_hat).cuda()
        rand_yz = th.cat([rand_y, rand_z], dim=1)
        s_middle, _ = self.inn_2(x=rand_yz, reverse=True)
        x_hat, _ = self.inn(x=s_middle, reverse=True)

        x_mmd = th.mean(mmd_matrix_multiscale(uf, x_hat, self.mmd_weight))

        loss = forward_loss + x_mmd * self.lamb

        loss.backward()
        th.nn.utils.clip_grad_norm_(self.trainable, 10.)
        self.opt.step()
        self.opt.zero_grad()

        if step % 50 == 0:
            writer.add_scalar('train/loss', loss, step)
            writer.add_scalar('train/middle_loss', middle_loss, step)
            writer.add_scalar('train/x_mmd', x_mmd, step)
            writer.add_scalar('train/cls_loss', cls_loss, step)
            writer.add_scalar('train/jac_loss', jac_loss, step)
            writer.add_scalar('train/z_loss', z_loss, step)
            # writer.add_scalar('train/loss_cls_new', loss_cls_new, step)
            # writer.add_scalar('train/err', err, step)
            print('step {} loss {}'.format(step, loss.item()))
        return cls_emb

    def _hook(self, writer: SummaryWriter, step):
        seen_data = self.reader.get_batch_tensor(self.reader.parts[1])
        seen_feat = seen_data[0]
        seen_label = seen_data[1]
        cls_emb = seen_data[5]

        unseen_data = self.reader.get_batch_tensor(self.reader.parts[2])
        unseen_feat = unseen_data[0]
        unseen_label = unseen_data[1]
        with th.no_grad():
            zeros = th.zeros([self.cls_num, self.feat_size - self.emb_size]).cuda()
            cls_zeros = th.cat([cls_emb, zeros], dim=1)
            # cls_emb = cls_emb.cpu().numpy()
            cls_s, _ = self.inn_2(x=cls_zeros, reverse=True)
            cls_s = cls_s.cpu().numpy()

            seen_cls_v, _ = self.inn(x=seen_feat)
            unseen_cls_v, _ = self.inn(x=unseen_feat)

            seen_acc = zsl_acc.cls_wise_acc(seen_cls_v.cpu().numpy(), seen_label.cpu().numpy(), cls_s)

            unseen_acc = zsl_acc.cls_wise_acc(unseen_cls_v.cpu().numpy(), unseen_label.cpu().numpy(), cls_s)

            h_score = 2 * (seen_acc * unseen_acc) / (seen_acc + unseen_acc)

            writer.add_scalar('hook/seen_acc', seen_acc, step)
            writer.add_scalar('hook/unseen_acc', unseen_acc, step)
            writer.add_scalar('hook/h_score', h_score, step)

    def train(self, task='hehe1', max_iter=50000):
        scheduler = th.optim.lr_scheduler.MultiStepLR(self.opt, milestones=[20, 40], gamma=0.1)
        time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
        writer_name = os.path.join(general.ROOT_PATH + 'result/{}/log/'.format(self.set_name), task + time_string)
        writer = SummaryWriter(writer_name)

        for i in range(max_iter):
            self._step(writer, i)
            if i % 50 == 0:
                self._hook(writer, i)

            if i % 1000 == 0 and i > 0:
                # noinspection PyArgumentList
                scheduler.step()


if __name__ == '__main__':
    settings = {'task_name': 'mix_1',
                'set_name': 'AWA1',
                'lamb': 1.,
                'lr': 5e-4,
                'depth': 5,
                'mmd_weight': [(0.1, 1), (0.2, 1), (1.5, 1), (3.0, 1), (5.0, 1), (10.0, 1)],
                'batch_size': 256,
                'max_iter': 50000}
    model = MixedTrainable(**settings)
    model.train(task=settings['task_name'], max_iter=settings['max_iter'])
