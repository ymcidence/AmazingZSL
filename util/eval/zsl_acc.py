import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def cls_wise_acc(img_feat: np.ndarray, label: np.ndarray, emb: np.ndarray):
    """

    :param img_feat: [N D]
    :param label: [N C]
    :param emb: [C D]
    :return:
    """
    cls_num = label.shape[1]
    distances = euclidean_distances(img_feat, emb)
    prediction = np.argmin(distances, axis=1)
    gt = np.argmax(label, axis=1)
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
    cls_num = label.shape[1]
    prediction = np.argmax(logits, axis=1)
    gt = np.argmax(label, axis=1)
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
