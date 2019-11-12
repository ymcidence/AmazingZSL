import FrEIA.framework as ff
import FrEIA.modules as fm
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
import matplotlib.pyplot as plt
import PIL.Image as Image


def fig2data(fig):
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tostring())


def draw(points):
    figure = plt.figure()
    plot = figure.add_subplot(111)
    x = points[:, 0]
    y = points[:, 1]
    plot.scatter(x, y)
    plt.savefig("./my_img.png")
    rslt = fig2img(figure)
    plt.close()
    return np.array(rslt)


class ToyModule(nn.Module):
    def __init__(self, lr=5e-4, feat_length=4, depth=2):
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
            return nn.Sequential(nn.Linear(ch_in, 2),
                                 nn.ReLU(),
                                 nn.Linear(2, ch_out))

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


class ToyTrainable(object):
    def __init__(self, **kwargs):
        self.set_name = kwargs.get('set_name', 'AWA1')
        self.lamb = kwargs.get('lamb', 3)
        self.lr = kwargs.get('lr', 5e-4)
        self.depth = kwargs.get('depth', 2)
        self.mmd_weight = kwargs.get('mmd_weight', [(0.1, 1), (0.2, 1), (1.5, 1), (3.0, 1), (5.0, 1), (10.0, 1)])
        self.seen_num = 3
        self.unseen_num = 1
        self.cls_num = self.seen_num + self.unseen_num
        self.feat_size = 4
        self.emb_size = 2
        self.batch_size = kwargs.get('batch_size', 256)
        self.reader = Reader(set_name=self.set_name, batch_size=self.batch_size, pre_process=True)
        # noinspection PyUnresolvedReferences
        self.inn = ToyModule(self.lr, feat_length=self.feat_size, depth=self.depth).cuda()

    def _step(self, writer: SummaryWriter, step):
        batch_data = self.reader.get_batch_tensor(self.reader.parts[0])
        feat = batch_data[0]
        label = batch_data[1]
        label_emb = batch_data[2]
        s_cls = [0, 1, 2]
        u_cls = [3]
        cls_emb = batch_data[5]

        ud = self.reader.get_batch_tensor(self.reader.parts[2])
        uf = ud[0]
        ul = ud[1]

        # 1. forward
        yz_hat, yz_det = self.inn(x=feat)
        y_hat = yz_hat[:, :self.emb_size]
        z_hat = yz_hat[:, self.emb_size:]

        cls_loss = th.mean((y_hat - label_emb) ** 2)
        jac_loss = -1 * th.mean(yz_det) / self.feat_size
        z_loss = th.mean(z_hat ** 2) / 2

        rand_y = cls_emb[[3] * self.batch_size, :]
        rand_z = th.randn_like(z_hat).cuda()
        rand_yz = th.cat([rand_y, rand_z], dim=1)
        x_hat, _ = self.inn(x=rand_yz, reverse=True)

        rand_yz_2 = th.cat([label_emb, rand_z], dim=1)
        x_hat_2, _ = self.inn(x=rand_yz_2, reverse=True)

        # x_loss = th.mean(x_hat[:, 2:] ** 2, dim=[0, 1]) + th.mean(x_hat_2[:, 2:] ** 2, dim=[0, 1])

        x_loss = th.mean(mmd_matrix_multiscale(feat, x_hat_2, self.mmd_weight))

        x_mmd = 0 * th.mean(mmd_matrix_multiscale(feat, x_hat, self.mmd_weight))

        loss = 10 * cls_loss + jac_loss + z_loss + x_mmd + 0 * x_loss

        loss.backward()
        th.nn.utils.clip_grad_norm_(self.inn.trainable_parameters, 10.)
        self.inn.optimizer.step()
        self.inn.optimizer.zero_grad()

        if step % 50 == 0:
            writer.add_scalar('train/loss', loss, step)
            writer.add_scalar('train/x_mmd', x_mmd, step)
            writer.add_scalar('train/cls_loss', cls_loss, step)
            writer.add_scalar('train/jac_loss', jac_loss, step)
            writer.add_scalar('train/z_loss', z_loss, step)
            writer.add_scalar('train/x_loss2', x_loss, step)
            # writer.add_scalar('train/loss_cls_new', loss_cls_new, step)
            # writer.add_scalar('train/err', err, step)
            print('step {} loss {}'.format(step, loss.item()))
        return cls_emb

    def _hook(self, writer: SummaryWriter, step):
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
        rand_z = th.randn_like(x[:, self.emb_size:]).cuda()
        rand_yz = th.cat([emb, rand_z], dim=1)
        gen, _ = self.inn(x=rand_yz, reverse=True)
        return gen

    def train(self, task='hehe1', max_iter=2000):
        scheduler = th.optim.lr_scheduler.MultiStepLR(self.inn.optimizer, milestones=[20, 40], gamma=0.1)
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
                th.save(self.inn.state_dict(), general.ROOT_PATH + 'result/{}/model/hehe.pt'.format(self.set_name))

        s_gen = []
        u_gen = []
        s_label = []
        u_label = []
        s_f = []
        u_f = []
        for i in range(max_iter, max_iter+10):
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

        sio.savemat(os.path.join(general.ROOT_PATH + 'result/{}/{}.mat'.format(self.set_name, task)), save_dict)


if __name__ == '__main__':
    settings = {'task_name': 'toy_under2',
                'set_name': 'Toy',
                'lamb': 1.,
                'lr': 5e-3,
                'depth': 3,
                'mmd_weight': [(0.1, 1), (0.2, 1), (1.5, 1), (3.0, 1), (5.0, 1), (10.0, 1)],
                'batch_size': 128 * 3,
                'max_iter': 4000}
    model = ToyTrainable(**settings)
    model.train(task=settings['task_name'], max_iter=settings['max_iter'])
