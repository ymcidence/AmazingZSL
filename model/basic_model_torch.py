import FrEIA.framework as ff
import FrEIA.modules as fm
import torch.nn as nn
import torch as th
from util.data import set_profiles
from util.data.dataset import ZSLArrayReader as Reader


class BasicModule(nn.Module):
    def __init__(self, lr=5e-4, feat_length=2048, depth=20):
        super().__init__()
        self.lr = lr
        self.feat_length = feat_length
        self.depth = depth

        self.trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        for p in self.trainable_parameters:
            # noinspection PyUnresolvedReferences
            p.data = 0.01 * th.randn_like(p)

        self.optimizer = th.optim.Adam(self.trainable_parameters, lr=lr, weight_decay=1e-5)

    def _inn(self):
        def subnet(ch_in, ch_out):
            return nn.Sequential(nn.Linear(ch_in, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, ch_out))

        nodes = []
        nodes.append(ff.InputNode(self.feat_length))
        for k in range(self.depth):
            nodes.append(ff.Node(nodes[-1], fm.PermuteRandom, {'seed': k}))
            nodes.append(ff.Node(nodes[-1], fm.GLOWCouplingBlock,
                                 {'subnet_constructor': subnet, 'clamp': 2.0}))

        return ff.ReversibleGraphNet(nodes + [ff.OutputNode(nodes[-1])], verbose=False)

    def forward(self, x, reverse=False):
        z = self.cinn(x, rev=reverse)
        jac = self.cinn.log_jacobian(rev=reverse, run_forward=False)
        return z, jac


class BasicTrainable(object):
    def __init__(self, **kwargs):
        self.set_name = kwargs.get('set_name', 'AWA1')
        self.lamb = kwargs.get('lamb', 3)
        self.lr = kwargs.get('lr', 5e-4)
        self.depth = kwargs.get('depth', 20)
        self.seen_num = set_profiles.LABEL_NUM[self.set_name][0]
        self.unseen_num = set_profiles.LABEL_NUM[self.set_name][1]
        self.cls_num = self.seen_num + self.unseen_num
        self.batch_size = kwargs.get('batch_size', 256)
        self.reader = Reader(set_name=self.set_name, batch_size=self.batch_size)
        self.inn = BasicModule(self.lr, depth=self.depth)

    def _build_tensor(self, part):
        this_feat = self.reader.get_batch_tensor()
