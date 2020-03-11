import torch.nn as nn
import torch as th
import numpy as np
import os
import general
from torch.utils.tensorboard import SummaryWriter
from util.data.toy_dataset import ToyReader as Reader
from util.layer.mmd import mmd_matrix_multiscale
from time import gmtime, strftime
from scipy import io as sio
from model.toy_model import draw

MSE = nn.MSELoss('mean')


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc_1 = nn.Sequential(nn.Linear(4, 4),
                                  nn.ReLU())
        self.mu = nn.Linear(4, 2)
        self.var = nn.Linear(4, 2)

    def forward(self, x, c):
        xc = th.cat((x, c), dim=-1)

        xc = self.fc_1(xc)

        means = self.mu(xc)
        log_vars = self.var(xc)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(nn.Linear(4, 4),
                                nn.ReLU(),
                                nn.Linear(4, 2))

    def forward(self, z, c):
        zc = th.cat((z, c), dim=-1)

        zc = self.fc(zc)

        return zc


class ToyCVAE(nn.Module):
    def __init__(self):
        super().__init__()
        # self.optimizer = th.optim.Adam(self.trainable_parameters, lr=1e-4, weight_decay=1e-5)
        self.enc = Encoder()
        self.dec = Decoder()
        scope = list(self.enc.parameters()) + list(self.dec.parameters())
        self.opt = th.optim.Adam(scope, lr=1e-4, weight_decay=1e-5)

    def forward(self, c, x=None, u_c=None):
        if x is not None and u_c is not None:
            mu, var = self.enc(x, c)
            batch_size = x.size(0)

            std = th.exp(0.5 * var)
            eps = th.randn([batch_size, 2]).cuda()
            z = eps * std + mu

            x_hat = self.dec(z, c)
            u_hat = self.inference(u_c)
            return x_hat, u_hat, mu, var, z
        else:
            return self.inference(c)

    def inference(self, c):
        batch_size = c.size(0)
        z = th.randn([batch_size, 2]).cuda()

        x_hat = self.dec(z, c)

        return x_hat


def cvae_loss(x_hat, x, mu, var):
    likelihood = MSE(x_hat, x)
    kld = -0.5 * th.sum(1 + var - mu.pow(2) - var.exp()) / x.size(0)

    return likelihood + kld


class ToyCVAETrainable(object):
    def __init__(self):
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.model = ToyCVAE().to(self.device)
        self.seen_num = 3
        self.unseen_num = 1
        self.cls_num = self.seen_num + self.unseen_num
        self.feat_size = 4
        self.emb_size = 2
        self.batch_size = 128 * 3
        self.mmd_weight = [(0.1, 1), (0.2, 1), (1.5, 1), (3.0, 1), (5.0, 1), (10.0, 1)]
        self.reader = Reader(set_name='Toy', batch_size=self.batch_size, pre_process=True, padding=False)

    def _step(self, writer: SummaryWriter, step):
        self.model.train()
        self.model.zero_grad()
        s_data = self.reader.get_batch_tensor(self.reader.parts[0])
        s_feat = s_data[0]
        s_label = s_data[1]
        s_label_emb = s_data[2]
        s_cls = [0, 1, 2]
        u_cls = [3]
        cls_emb = s_data[5]

        # u_data = self.reader.get_batch_tensor(self.reader.parts[2])
        # u_feat = u_data[0]
        # u_label = u_data[1]

        rand_z = th.randn_like(s_feat).cuda()
        s_cz = th.cat([s_label_emb, rand_z], dim=1)

        u_label_emb = cls_emb[u_cls * self.batch_size, :]

        x_hat, u_hat, mu, var, z = self.model(s_label_emb, s_feat, u_label_emb)

        # u_hat = self.model(u_cz)

        loss = cvae_loss(x_hat, s_feat, mu, var) - 0.1 * th.mean(mmd_matrix_multiscale(s_feat, u_hat, self.mmd_weight))
        loss.backward()
        self.model.opt.step()
        self.model.opt.zero_grad()

        if step % 50 == 0:
            writer.add_scalar('train/loss', loss, step)
            # writer.add_scalar('train/critic_loss', _critic_loss, step)

            # writer.add_scalar('train/loss_cls_new', loss_cls_new, step)
            # writer.add_scalar('train/err', err, step)
            print('step {} loss {}'.format(step, loss.item()))
        return cls_emb

    def train(self, task='hehe1', max_iter=10000):
        time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
        writer_name = os.path.join(general.ROOT_PATH + '/result/{}/log/'.format('ToyCVAE'), task + time_string)
        writer = SummaryWriter(writer_name)

        for i in range(max_iter):
            self._step(writer, i)
            if i % 50 == 0:
                self._hook(writer, i)
            if i % 1000 == 0 and i > 0:
                # noinspection PyArgumentList
                # scheduler.step()
                th.save(self.model.state_dict(), general.ROOT_PATH + '/result/{}/model/hehe.pt'.format('ToyCVAE'))

        s_gen = []
        u_gen = []
        s_label = []
        u_label = []
        s_f = []
        u_f = []
        for i in range(max_iter, max_iter + 10):
            outs = self._hook(writer, i)
            s_gen.append(outs[0])
            u_gen.append(outs[1])
            s_f.append(outs[2])
            u_f.append(outs[3])
            s_label.append(outs[4])
            u_label.append(outs[5])

        s_gen = np.concatenate(s_gen, axis=0)
        u_gen = np.concatenate(u_gen, axis=0)
        s_f = np.concatenate(s_f, axis=0)
        u_f = np.concatenate(u_f, axis=0)
        s_label = np.concatenate(s_label, axis=0)
        u_label = np.concatenate(u_label, axis=0)
        save_dict = {'s_gen': s_gen, 'u_gen': u_gen, 's_f': s_f, 'u_f': u_f, 's_label': s_label, 'u_label': u_label}

        sio.savemat(os.path.join(general.ROOT_PATH + '/result/{}/{}.mat'.format('ToyCVAE', task)), save_dict)

    def _hook(self, writer: SummaryWriter, step):
        self.model.eval()
        seen_data = self.reader.get_batch_tensor(self.reader.parts[1])
        seen_feat = seen_data[0]
        seen_label = seen_data[1]
        seen_le = seen_data[2]

        unseen_data = self.reader.get_batch_tensor(self.reader.parts[2])
        unseen_feat = unseen_data[0]
        unseen_label = unseen_data[1]
        unseen_le = unseen_data[2]

        with th.no_grad():
            seen_gen = self.generation(seen_feat, seen_le).cpu().numpy()
            unseen_gen = self.generation(unseen_feat, unseen_le).cpu().numpy()

            c_1 = np.mean(unseen_gen[:, 0])
            c_2 = np.mean(unseen_gen[:, 1])

            writer.add_scalar('hook_unseen/c_1', c_1, step)
            writer.add_scalar('hook_unseen/c_2', c_2, step)

            writer.add_image('hook/seen', draw(seen_gen), step, dataformats='HWC')
            writer.add_image('hook/unseen', draw(unseen_gen), step, dataformats='HWC')

        return seen_gen, unseen_gen, seen_feat.cpu().numpy(), unseen_feat.cpu().numpy(), seen_label.cpu().numpy(), unseen_label.cpu().numpy()

    def generation(self, x, emb):
        gen = self.model(emb)
        return gen


if __name__ == '__main__':
    settings = {'task_name': 'small_mmd',
                'set_name': 'Toy',
                'lamb': 1.,
                'lr': 5e-3,
                'depth': 3,
                'mmd_weight': [(0.1, 1), (0.2, 1), (1.5, 1), (3.0, 1), (5.0, 1), (10.0, 1)],
                'batch_size': 128 * 3,
                'max_iter': 10000}
    model = ToyCVAETrainable()
    model.train(task=settings['task_name'], max_iter=settings['max_iter'])
