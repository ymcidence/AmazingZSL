import torch as th
import torch.nn as nn
import FrEIA.framework as ff
import FrEIA.modules as fm
from torch.utils.tensorboard import SummaryWriter
from util.data.mnist import MNISTArrayReader as Reader
from time import gmtime, strftime
from util.layer.mmd import mmd_matrix_multiscale


def one_hot(labels, out=None):
    """
    Convert LongTensor labels (contains labels 0-9), to a one hot vector.
    Can be done in-place using the out-argument (faster, re-use of GPU memory)
    :param labels:
    :param out:
    :return:
    """
    if out is None:
        # noinspection PyUnresolvedReferences
        out = th.zeros(labels.shape[0], 10).to(labels.device)
    else:
        out.zeros_()

    out.scatter_(dim=1, index=labels.view(-1, 1), value=1.)
    return out


# noinspection PyUnresolvedReferences
class MNISTModel(nn.Module):
    def __init__(self, lr):
        super().__init__()

        self.cinn = self.build_inn()

        self.trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        for p in self.trainable_parameters:
            # noinspection PyUnresolvedReferences
            p.data = 0.01 * th.randn_like(p)

        self.optimizer = th.optim.Adam(self.trainable_parameters, lr=lr, weight_decay=1e-5)

    def build_inn(self):
        def subnet(ch_in, ch_out):
            return nn.Sequential(nn.Linear(ch_in, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, ch_out))

        cond = ff.ConditionNode(10)
        nodes = []
        nodes.append(ff.InputNode(1, 28, 28))

        nodes.append(ff.Node(nodes[-1], fm.Flatten, {}))

        for k in range(20):
            nodes.append(ff.Node(nodes[-1], fm.PermuteRandom, {'seed': k}))
            nodes.append(ff.Node(nodes[-1], fm.GLOWCouplingBlock,
                                 {'subnet_constructor': subnet, 'clamp': 1.0},
                                 conditions=cond))

        return ff.ReversibleGraphNet(nodes + [cond, ff.OutputNode(nodes[-1])], verbose=False)

    def forward(self, x, l, one_hot_l=False, reverse=False):
        l = l if one_hot_l else one_hot(l)
        z = self.cinn(x, c=l, rev=reverse)
        jac = self.cinn.log_jacobian(rev=reverse, run_forward=False)
        return z, jac


# noinspection PyUnresolvedReferences
class MNISTUnconModel(nn.Module):
    def __init__(self, lr):
        super().__init__()

        self.cinn = self.build_inn()

        self.trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        for p in self.trainable_parameters:
            # noinspection PyUnresolvedReferences
            p.data = 0.01 * th.randn_like(p)

        self.optimizer = th.optim.Adam(self.trainable_parameters, lr=lr, weight_decay=1e-5)

    def build_inn(self):
        def subnet(ch_in, ch_out):
            return nn.Sequential(nn.Linear(ch_in, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, ch_out))

        nodes = []
        nodes.append(ff.InputNode(28 * 28))

        nodes.append(ff.Node(nodes[-1], fm.Flatten, {}))

        for k in range(20):
            nodes.append(ff.Node(nodes[-1], fm.PermuteRandom, {'seed': k}))
            nodes.append(ff.Node(nodes[-1], fm.GLOWCouplingBlock,
                                 {'subnet_constructor': subnet, 'clamp': 1.0}))

        return ff.ReversibleGraphNet(nodes + [ff.OutputNode(nodes[-1])], verbose=False)

    def forward(self, x, l, one_hot_l=False, reverse=False):
        l = l if one_hot_l else one_hot(l)
        # z = self.cinn(x, c=l, rev=reverse)
        z = self.cinn(x, rev=reverse)
        jac = self.cinn.log_jacobian(rev=reverse, run_forward=False)
        return z, jac


def train(max_iter=50000):
    reader = Reader()
    model = MNISTModel(5e-4)
    # noinspection PyUnresolvedReferences
    model = model.cuda()
    scheduler = th.optim.lr_scheduler.MultiStepLR(model.optimizer, milestones=[20, 40], gamma=0.1)
    time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
    writer = SummaryWriter('./result/mnist2/log/' + 'no_cond' + time_string)
    for i in range(max_iter):
        x, l = reader.get_batch_tensor()
        x = reader.augmentation(x)

        z, log_j = model(x, l, one_hot_l=True)

        nll = th.mean(z ** 2) / 2 - th.mean(log_j) / (28 * 28)
        nll.backward()
        th.nn.utils.clip_grad_norm_(model.trainable_parameters, 10.)
        model.optimizer.step()
        model.optimizer.zero_grad()

        if i % 50 == 0:
            print('step {} loss {} lr {}'.format(i, nll.item(), model.optimizer.param_groups[0]['lr']))
            writer.add_scalar('loss', nll, i)
            img = x.view([-1, 28, 28, 1])
            writer.add_images('image', img[:16, :, :, :], i, dataformats='NHWC')
            z = 1.0 * th.randn(reader.batch_size, 28 * 28).cuda()
            with th.no_grad():
                samples, _ = model(z, l, one_hot_l=True, reverse=True)
                samples = samples * 0.305 + 0.128
                samples = th.clamp(samples, 0, 1)

                samples = samples.view([-1, 28, 28, 1])
                writer.add_images('gen', samples[:16, :, :, :], i, dataformats='NHWC')

        if i % 230 == 0 and i > 0:
            scheduler.step()
    writer.close()


def train_alt(max_iter=50000):
    reader = Reader()
    model = MNISTUnconModel(5e-4)
    # noinspection PyUnresolvedReferences
    model = model.cuda()
    scheduler = th.optim.lr_scheduler.MultiStepLR(model.optimizer, milestones=[20, 40], gamma=0.1)
    time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
    writer = SummaryWriter('./model/result/mnist2/log/' + 'old_cond_only_det' + time_string)
    for i in range(max_iter):
        x, l = reader.get_batch_tensor()
        x = reader.augmentation(x)

        z0, log_j = model(x, l, one_hot_l=True)

        y = z0[:, :10]
        z = z0[:, 10:]

        loss_y = th.mean((y - l) ** 2)

        random_signal = th.randn_like(z).cuda()

        z_hat = th.cat([l, random_signal], dim=1)

        mmd_y = mmd_matrix_multiscale(z0, z_hat,
                                      widths_exponents=[(0.2, 2), (1.5, 2), (3.0, 2)])

        x_hat, _ = model(z_hat, l, one_hot_l=True, reverse=True)

        mmd_x = mmd_matrix_multiscale(x, x_hat,
                                      widths_exponents=[(0.2, 2), (1.5, 2), (3.0, 2)])

        loss = loss_y + (th.mean(mmd_y) + th.mean(mmd_x)) * 0. + th.mean(z ** 2) / 2 - th.mean(log_j) / (28 * 28)



        loss.backward()
        th.nn.utils.clip_grad_norm_(model.trainable_parameters, 10.)
        model.optimizer.step()
        model.optimizer.zero_grad()

        if i % 50 == 0:
            print('step {} loss {} lr {}'.format(i, loss.item(), model.optimizer.param_groups[0]['lr']))
            writer.add_scalar('loss', loss, i)
            img = x.view([-1, 28, 28, 1])
            writer.add_images('image', img[:16, :, :, :], i, dataformats='NHWC')
            zz = 1.0 * th.randn(reader.batch_size, 28 * 28 - 10).cuda()
            with th.no_grad():
                zz = th.cat([l, zz], dim=1)
                samples, _ = model(zz, l, one_hot_l=True, reverse=True)
                samples = samples * 0.305 + 0.128
                samples = th.clamp(samples, 0, 1)

                samples = samples.view([-1, 28, 28, 1])
                writer.add_images('gen', samples[:16, :, :, :], i, dataformats='NHWC')

        if i % 230 == 0 and i > 0:
            scheduler.step()
    writer.close()


if __name__ == '__main__':
    train_alt()
