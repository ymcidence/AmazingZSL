import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import torch as th


def one_hot(labels, out_dim=10):
    """
    Convert LongTensor labels (contains labels 0-out_dim), to a one hot vector.
    Can be done in-place using the out-argument (faster, re-use of GPU memory)
    :param labels:
    :param out_dim:
    :return:
    """
    # noinspection PyUnresolvedReferences
    out = th.zeros(labels.shape[0], out_dim).to(labels.device)

    out.scatter_(dim=1, index=labels.view(-1, 1), value=1.)
    return out


def cls_wise_acc(img_feat: np.ndarray, label: np.ndarray, emb: np.ndarray, type='euclidean'):
    """

    :param img_feat: [N D]
    :param label: [N C]
    :param emb: [C D]
    :param type:
    :return:
    """
    if type == 'euclidean':
        distances = euclidean_distances(img_feat, emb)
    else:
        distances = cosine_similarity(img_feat, emb) * -1
    prediction = np.argmin(distances, axis=1)
    if label.shape.__len__() > 1:
        cls_num = label.shape[1]
        gt = np.argmax(label, axis=1)
    else:
        cls_num = int(np.max(label) + 1)
        gt = label.astype(np.int32)
    acc = []
    for i in range(cls_num):
        ind = np.where(gt == i)[0]

        if ind.shape[0] < 1:
            continue
        cls_pred = prediction[ind]
        correct_num = np.where(cls_pred == i)[0].shape[0]
        cls_acc = correct_num / ind.shape[0]

        acc.append(cls_acc)
    acc = np.asarray(acc)

    return np.sum(acc) / acc.shape[0] if acc.shape[0] > 0 else 0


def cls_wise_prob_acc(logits: np.ndarray, label: np.ndarray):
    prediction = np.argmax(logits, axis=1)
    if label.shape.__len__() > 1:
        cls_num = label.shape[1]
        gt = np.argmax(label, axis=1)
    else:
        cls_num = int(np.max(label) + 1)
        gt = label.astype(np.int32)
    acc = []
    for i in range(cls_num):
        ind = np.where(gt == i)[0]

        if ind.shape[0] < 1:
            continue
        cls_pred = prediction[ind]
        correct_num = np.where(cls_pred == i)[0].shape[0]
        cls_acc = correct_num / ind.shape[0]

        acc.append(cls_acc)
    acc = np.asarray(acc)

    return np.sum(acc) / acc.shape[0] if acc.shape[0] > 0 else 0


def h_score(seen_acc, unseen_acc):
    return 2 * seen_acc * unseen_acc / (seen_acc + unseen_acc + 1e-8)


def test():
    a = [[1.5, 1.5, 1.4],
         [1.1, 1.2, 1.2],
         [2.1, 2.5, 2.1],
         [2.1, 2.5, 2.1],
         [3.1, 3.5, 3.1],
         [3.3, 2.9, 3.1],
         [3.3, 2.9, 3.1],
         [3.3, 2.9, 3.1],
         [3.3, 2.9, 3.1]]
    a = np.asarray(a)
    b = [[1, 1, 1.],
         [2, 2, 2],
         [3, 3, 3]]
    b = np.asarray(b)
    label = [0, 0, 1, 1, 1, 2, 2, 2, 2]
    label = np.asarray(label)
    label = np.eye(3)[label]
    acc = cls_wise_acc(a, label, b)
    print(acc)


def test_2():
    a = [[0, 5, -4.],
         [2, 1, 0.],
         [4, 4, 9],
         [3, 1, -22]]
    a = np.asarray(a)
    b = [[0, 1, 0],
         [1, 0, 0],
         [0, 0, 1],
         [0, 0, 1]]
    b = np.asarray(b)
    label = b
    acc = cls_wise_prob_acc(a, label)
    print(acc)


if __name__ == '__main__':
    test_2()
