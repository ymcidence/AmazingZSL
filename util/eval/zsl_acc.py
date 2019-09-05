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
    acc = 0
    acc_num = 0
    for i in range(cls_num):
        ind = np.where(gt == i)[0]

        if ind.shape[0] < 1:
            continue
        cls_pred = prediction[ind]
        correct_num = np.where(cls_pred == i)[0].shape[0]
        cls_acc = correct_num / ind.shape[0]

        acc_num += 1
        acc += cls_acc

    return acc / acc_num


def h_score(seen_acc, unseen_acc):
    return 2 * seen_acc * unseen_acc / (seen_acc + unseen_acc + 1e-7)
