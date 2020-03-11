import FrEIA.framework as ff
import FrEIA.modules as fm
import torch.nn as nn
import torch as th
import numpy as np
import os
import general
from torch.utils.tensorboard import SummaryWriter
from util.data import set_profiles
from util.data.dataset import ZSLArrayReader as Reader
from util.layer.mmd import mmd_matrix_multiscale
from util.eval import zsl_acc
from time import gmtime, strftime
from scipy import io as sio


def cosine(a: th.Tensor, b: th.Tensor):
    a_b = th.matmul(a, b.transpose(1, 0))
    a_s = th.sqrt(th.sum(a ** 2, dim=1, keepdim=True))
    b_s = th.sqrt(th.sum(b ** 2, dim=1, keepdim=True)).transpose(1, 0)
    ab = th.matmul(a_s, b_s) + 1e-8

    return a_b / ab


class BasicModule(nn.Module):
    def __init__(self, lr=5e-4, feat_length=2048, depth=20):
        super().__init__()
        self.lr = lr
        self.feat_length = feat_length
        self.depth = depth
        self.cinn = self._inn()

        # noinspection PyUnresolvedReferences
        self.trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        for p in self.trainable_parameters:
            # noinspection PyUnresolvedReferences
            p.data = 0.01 * th.randn_like(p)

        self.optimizer = th.optim.Adam(self.trainable_parameters, lr=lr, weight_decay=1e-5)

    def _inn(self):
        def subnet(ch_in, ch_out):
            return nn.Sequential(nn.Linear(ch_in, 1024),
                                 nn.LeakyReLU(),
                                 nn.Linear(1024, ch_out))

        nodes = []
        nodes.append(ff.InputNode(self.feat_length))
        for k in range(self.depth):
            nodes.append(ff.Node(nodes[-1], fm.PermuteRandom, {'seed': k}))
            nodes.append(ff.Node(nodes[-1], fm.GLOWCouplingBlock,
                                 {'subnet_constructor': subnet, 'clamp': 2.0}))

        return ff.ReversibleGraphNet(nodes + [ff.OutputNode(nodes[-1])], verbose=False)

    # noinspection PyUnresolvedReferences
    def forward(self, x, reverse=False):
        z = self.cinn(x, rev=reverse)
        jac = self.cinn.log_jacobian(rev=reverse, run_forward=False)
        return z, jac


class LinearModule(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()

        self.fc = th.nn.Linear(size_in, size_out)

        self.ce = nn.CrossEntropyLoss()

        self.opt = th.optim.Adam(self.fc.parameters(), lr=5e-4)

    def forward(self, x):
        return self.fc(x)


class ClsTrainable(object):
    def __init__(self, **kwargs):
        self.set_name = kwargs.get('set_name', 'AWA1')
        self.lamb = kwargs.get('lamb', 3)
        self.lr = kwargs.get('lr', 5e-4)
        self.depth = kwargs.get('depth', 20)
        self.mmd_weight = kwargs.get('mmd_weight', [(0.1, 1), (0.2, 1), (1.5, 1), (3.0, 1), (5.0, 1), (10.0, 1)])
        self.seen_num = set_profiles.LABEL_NUM[self.set_name][0]
        self.unseen_num = set_profiles.LABEL_NUM[self.set_name][1]
        self.cls_num = self.seen_num + self.unseen_num
        self.feat_size = set_profiles.FEAT_DIM[self.set_name]
        self.emb_size = set_profiles.ATTR_DIM[self.set_name]
        self.batch_size = kwargs.get('batch_size', 256)
        self.reader = Reader(set_name=self.set_name, batch_size=self.batch_size, pre_process=True)
        # noinspection PyUnresolvedReferences
        self.inn = BasicModule(self.lr, feat_length=self.feat_size, depth=self.depth).cuda()

        # noinspection PyUnresolvedReferences
        self.cls = LinearModule(self.feat_size, self.cls_num).cuda()
        self.ce = nn.CrossEntropyLoss()
        self.dist = nn.CosineSimilarity()

    def _cls(self, writer: SummaryWriter, step):
        batch_data = self.reader.get_batch_tensor(self.reader.parts[0])
        feat = batch_data[0]
        label = batch_data[1]
        label_emb = batch_data[2]
        s_cls = batch_data[3]
        u_cls = batch_data[4]
        cls_emb = batch_data[5]
        # noise = th.rand_like(feat).cuda() * 0.008

        rand_ul_ind = np.random.randint(0, self.unseen_num, self.batch_size, dtype=np.int32)
        rand_ul = u_cls[rand_ul_ind]
        rand_y = cls_emb[rand_ul.long(), :]
        rand_z = th.randn([self.batch_size, self.feat_size - self.emb_size]).cuda()
        rand_yz = th.cat([rand_y, rand_z], dim=1).detach()
        x_hat, _ = self.inn(x=rand_yz, reverse=True)

        ll = th.cat([label, rand_ul], dim=0)
        xx = th.cat([feat, x_hat], dim=0)

        # target = zsl_acc.one_hot(ll.long(), self.cls_num).long()

        loss = self.ce(self.cls(x=xx), ll.long())

        loss.backward()
        # noinspection PyUnresolvedReferences
        self.cls.opt.step()
        # noinspection PyUnresolvedReferences
        self.cls.opt.zero_grad()
        if step % 50 == 0:
            writer.add_scalar('train/loss', loss, step)
            print('step {} (cls) loss {}'.format(step, loss.item()))

    def _step(self, writer: SummaryWriter, step):
        batch_data = self.reader.get_batch_tensor(self.reader.parts[0])
        feat = batch_data[0]
        label = batch_data[1]
        label_emb = batch_data[2]
        s_cls = batch_data[3]
        u_cls = batch_data[4]
        cls_emb = batch_data[5]
        noise = th.rand_like(feat).cuda() * 0.008

        ud = self.reader.get_batch_tensor(self.reader.parts[2])
        uf = ud[0]
        ul = ud[1]

        # 1. forward
        yz_hat, yz_det = self.inn(x=feat + noise)
        y_hat = yz_hat[:, :self.emb_size]
        z_hat = yz_hat[:, self.emb_size:]

        cls_loss = th.mean((y_hat - label_emb) ** 2)
        jac_loss = -1 * th.mean(yz_det) / self.feat_size
        z_loss = th.sum(z_hat ** 2) /2 #/ self.batch_size

        # 2. reverse
        rand_ul_ind = np.random.randint(0, self.unseen_num, self.batch_size, dtype=np.int32)
        rand_ul = u_cls[rand_ul_ind]
        rand_y = cls_emb[rand_ul.long(), :]
        rand_z = th.randn_like(z_hat).cuda()
        rand_yz = th.cat([rand_y, rand_z], dim=1)
        x_hat, _ = self.inn(x=rand_yz, reverse=True)

        x_mmd = th.mean(mmd_matrix_multiscale(uf, x_hat, self.mmd_weight))

        x_hat_2 = th.cat([cls_emb[label.long(), :], rand_z], dim=1)

        # yz_mmd = th.mean(mmd_matrix_multiscale(feat, x_hat_2, self.mmd_weight))
        yz_mmd = th.mean(mmd_matrix_multiscale(rand_z, z_hat, self.mmd_weight))

        # 3. reverse_2

        zero_padding = th.zeros([self.cls_num, self.feat_size - self.emb_size]).cuda()

        prototypes = th.cat([cls_emb, zero_padding], dim=1)

        x_p, _ = self.inn(x=prototypes, reverse=True)

        ff = th.cat([feat, x_hat.detach()], dim=0)
        ll = th.cat([label.long(), rand_ul.long()], dim=0)
        dist = cosine(ff, x_p)
        cls_loss_2 = self.ce(dist, ll)

        # 3. forward-rev

        # x_new = x_hat.detach()
        # yz_new, det_hat = self.inn(x=x_new)
        #
        # err = th.mean((yz_new - rand_yz) ** 2) / 2
        # y_new = yz_new[:, :self.emb_size]
        #
        # loss_cls_new = th.mean((y_new - rand_y) ** 2) / 2

        # FINAL. loss
        rr = 0 if step <= 500 else 0
        loss = 10 * cls_loss + rr * cls_loss_2 + jac_loss + z_loss + (yz_mmd + x_mmd) * self.lamb
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.inn.trainable_parameters, 10.)
        self.inn.optimizer.step()
        self.inn.optimizer.zero_grad()
        if step % 50 == 0:
            writer.add_scalar('train/loss', loss, step)
            writer.add_scalar('train/yz_mmd', yz_mmd, step)
            writer.add_scalar('train/x_mmd', x_mmd, step)
            writer.add_scalar('train/cls_loss', cls_loss, step)
            writer.add_scalar('train/jac_loss', jac_loss, step)
            writer.add_scalar('train/z_loss', z_loss, step)
            writer.add_scalar('train/cls_loss_2', cls_loss_2, step)
            # writer.add_scalar('train/loss_cls_new', loss_cls_new, step)
            # writer.add_scalar('train/err', err, step)
            print('step {} loss {}'.format(step, loss.item()))
        return cls_emb

    # hook

    def _hook_s(self, writer: SummaryWriter, step):
        seen_data = self.reader.get_batch_tensor(self.reader.parts[1])
        seen_feat = seen_data[0]
        seen_label = seen_data[1]
        cls_emb = seen_data[5]

        unseen_data = self.reader.get_batch_tensor(self.reader.parts[2])
        unseen_feat = unseen_data[0]
        unseen_label = unseen_data[1]
        with th.no_grad():
            cls_emb = cls_emb.cpu().numpy()
            seen_yz_hat, _ = self.inn(x=seen_feat)
            seen_y_hat = seen_yz_hat[:, :self.emb_size]
            seen_acc = zsl_acc.cls_wise_acc(seen_y_hat.cpu().numpy(), seen_label.cpu().numpy(), cls_emb)

            unseen_yz_hat, _ = self.inn(x=unseen_feat)
            unseen_y_hat = unseen_yz_hat[:, :self.emb_size]
            unseen_acc = zsl_acc.cls_wise_acc(unseen_y_hat.cpu().numpy(), unseen_label.cpu().numpy(), cls_emb)

            h_score = 2 * (seen_acc * unseen_acc) / (seen_acc + unseen_acc)

            writer.add_scalar('hook_s/seen_acc', seen_acc, step)
            writer.add_scalar('hook_s/unseen_acc', unseen_acc, step)
            writer.add_scalar('hook_s/h_score', h_score, step)

    def _hook_v(self, writer: SummaryWriter, step):
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
            cls_x, _ = self.inn(x=cls_zeros, reverse=True)
            cls_x = cls_x.cpu().numpy()
            seen_acc = zsl_acc.cls_wise_acc(seen_feat.cpu().numpy(), seen_label.cpu().numpy(), cls_x, 'euclidean')

            unseen_acc = zsl_acc.cls_wise_acc(unseen_feat.cpu().numpy(), unseen_label.cpu().numpy(), cls_x, 'euclidean')

            h_score = 2 * (seen_acc * unseen_acc) / (seen_acc + unseen_acc)

            writer.add_scalar('hook_v/seen_acc', seen_acc, step)
            writer.add_scalar('hook_v/unseen_acc', unseen_acc, step)
            writer.add_scalar('hook_v/h_score', h_score, step)

    def _hook_c(self, writer: SummaryWriter, step):
        seen_data = self.reader.get_batch_tensor(self.reader.parts[1])
        seen_feat = seen_data[0]
        seen_label = seen_data[1]
        seen_le = seen_data[2]

        unseen_data = self.reader.get_batch_tensor(self.reader.parts[2])
        unseen_feat = unseen_data[0]
        unseen_label = unseen_data[1]
        unseen_le = unseen_data[2]
        with th.no_grad():
            seen_pred = self.cls(x=seen_feat).cpu().numpy()
            unseen_pred = self.cls(x=unseen_feat).cpu().numpy()

            seen_acc = zsl_acc.cls_wise_prob_acc(seen_pred, seen_label.cpu().numpy())
            unseen_acc = zsl_acc.cls_wise_prob_acc(unseen_pred, unseen_label.cpu().numpy())

            seen_gen = self.generation(seen_feat, seen_le).cpu().numpy()
            unseen_gen = self.generation(unseen_feat, unseen_le).cpu().numpy()

            h_score = 2 * (seen_acc * unseen_acc) / (seen_acc + unseen_acc)

            writer.add_scalar('hook_c/seen_acc', seen_acc, step)
            writer.add_scalar('hook_c/unseen_acc', unseen_acc, step)
            writer.add_scalar('hook_c/h_score', h_score, step)

        return seen_gen, unseen_gen, unseen_pred, seen_label.cpu().numpy(), unseen_label.cpu().numpy()

    def generation(self, x, emb):
        rand_z = th.randn_like(x[:, self.emb_size:]).cuda()
        rand_yz = th.cat([emb, rand_z], dim=1)
        gen, _ = self.inn(x=rand_yz, reverse=True)
        return gen

    def train(self, task='hehe1', max_iter=50000):
        scheduler = th.optim.lr_scheduler.MultiStepLR(self.inn.optimizer, milestones=[20, 40], gamma=0.1)
        time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
        writer_name = os.path.join(general.ROOT_PATH + 'result/{}/log/'.format(self.set_name), task + time_string)
        writer = SummaryWriter(writer_name)

        for i in range(11000):
            self._step(writer, i)
            if i % 50 == 0:
                self._hook_v(writer, i)
                self._hook_s(writer, i)

            if i % 1000 == 0 and i > 0:
                # noinspection PyArgumentList
                scheduler.step()
                th.save(self.inn.state_dict(), general.ROOT_PATH + 'result/{}/model/hehe.pt'.format(self.set_name))

        s_gen = []
        u_gen = []
        s_label = []
        u_label = []
        pred = []
        for i in range(11000, 12000):
            self._cls(writer, i)
            if i % 50 == 0:
                outs = self._hook_c(writer, i)
                s_gen.append(outs[0])
                u_gen.append(outs[1])
                pred.append(outs[2])
                s_label.append(outs[3])
                u_label.append(outs[4])

        s_gen = np.concatenate(s_gen, axis=0)
        u_gen = np.concatenate(u_gen, axis=0)
        pred = np.concatenate(pred, axis=0)
        s_label = np.concatenate(s_label, axis=0)
        u_label = np.concatenate(u_label, axis=0)

        save_dict = {'s_gen': s_gen, 'u_gen': u_gen, 'pred': pred, 's_label': s_label, 'u_label': u_label}

        sio.savemat(os.path.join(general.ROOT_PATH + 'result/{}/hehe.mat'.format(self.set_name)), save_dict)


if __name__ == '__main__':
    settings = {'task_name': 'cosine',
                'set_name': 'APY',
                'lamb': 1.,
                'lr': 5e-4,
                'depth': 5,
                'mmd_weight': [(1.5, 1), (3.0, 1), (5.0, 1), (10.0, 1)],
                'batch_size': 256,
                'max_iter': 50000}
    model = ClsTrainable(**settings)
    model.train(task=settings['task_name'], max_iter=settings['max_iter'])