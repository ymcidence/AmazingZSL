import torch as th
import numpy as np


def batch_distance(x: th.Tensor):
    xx = th.mm(x, x.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    return th.clamp(rx.t() + rx - 2. * xx, 0, np.inf)


def batch_adjacency(x: th.Tensor, scale=.1):
    dx = batch_distance(x)
    return th.exp(-1 * dx / scale)


def test_adj():
    a = th.tensor([[1, 2, 3], [2, 3, 5]], dtype=th.float32)

    b = build_adjacency(a)

    c = th.exp(-1 * b / .1) - th.eye(2)
    print(c)


if __name__ == '__main__':
    test_adj()
