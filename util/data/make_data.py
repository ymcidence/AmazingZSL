import os
import tensorflow as tf
import numpy as np
import scipy.io as sio
from util.data import set_profiles


def get_mat_name(set_name, root_folder):
    split_file = os.path.join(root_folder, set_name, 'att_splits.mat')
    feat_file = os.path.join(root_folder, set_name, 'res101.mat')

    return split_file, feat_file


def process_mat(set_name, root_folder):
    split_file, feat_file = get_mat_name(set_name, root_folder)
    split_mat = sio.loadmat(split_file)
    feat_mat = sio.loadmat(feat_file)

    training_ind = np.squeeze(split_mat['trainval_loc'] - 1)
    training_feat = feat_mat['features'][:, training_ind].T
    training_label = np.squeeze(feat_mat['labels'][training_ind] - 1)
    training_label_emb = split_mat['att'][:, training_label].T
    training_label_set = np.sort(np.unique(training_label), 0)
    assert training_label_set.__len__() == set_profiles.LABEL_NUM.get(set_name)[0]

    seen_ind = np.squeeze(split_mat['test_seen_loc'] - 1)
    seen_feat = feat_mat['features'][:, seen_ind].T
    seen_label = np.squeeze(feat_mat['labels'][seen_ind] - 1)
    seen_label_emb = split_mat['att'][:, seen_label].T
    seen_label_set = np.sort(np.unique(seen_label), 0)
    assert seen_label_set.__len__() == set_profiles.LABEL_NUM.get(set_name)[0]

    unseen_ind = np.squeeze(split_mat['test_unseen_loc'] - 1)
    unseen_feat = feat_mat['features'][:, unseen_ind].T
    unseen_label = np.squeeze(feat_mat['labels'][unseen_ind] - 1)
    unseen_label_emb = split_mat['att'][:, unseen_label].T
    unseen_label_set = np.sort(np.unique(unseen_label), 0)
    assert unseen_label_set.__len__() == set_profiles.LABEL_NUM.get(set_name)[1]

    seen_cls_emb = split_mat['att'][:, seen_label_set].T
    unseen_cls_emb = split_mat['att'][:, unseen_label_set].T
    cls_emb = split_mat['att'].T

    training_dict = {'feat': training_feat,
                     'id': training_ind,
                     'label': training_label,
                     'label_emb': training_label_emb,
                     's_cls': training_label_set,
                     'u_cls': unseen_label_set,
                     'cls_emb': cls_emb,
                     's_cls_emb': seen_cls_emb,
                     'u_cls_emb': unseen_cls_emb}

    seen_dict = {'feat': seen_feat,
                 'id': seen_ind,
                 'label': seen_label,
                 'label_emb': seen_label_emb,
                 's_cls': training_label_set,
                 'u_cls': unseen_label_set,
                 'cls_emb': cls_emb,
                 's_cls_emb': seen_cls_emb,
                 'u_cls_emb': unseen_cls_emb}

    unseen_dict = {'feat': unseen_feat,
                   'id': unseen_ind,
                   'label': unseen_label,
                   'label_emb': unseen_label_emb,
                   's_cls': training_label_set,
                   'u_cls': unseen_label_set,
                   'cls_emb': cls_emb,
                   's_cls_emb': seen_cls_emb,
                   'u_cls_emb': unseen_cls_emb}
    return training_dict, seen_dict, unseen_dict


def _int64_feature(value):
    """Create a feature that is serialized as an int64."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def convert_tfrecord(data, set_name, part_name):
    data_length = data['feat'].shape[0]

    save_path = os.path.join('./data', set_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_name = os.path.join(save_path, part_name + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(file_name)

    for i in range(data_length):
        print(i)
        this_id = _int64_feature(data['id'][i])
        this_feat = _float_feature(data['feat'][i, :])
        this_label = _int64_feature(data['label'][i])
        this_label_emb = _float_feature(data['label_emb'][i, :])
        feat_dict = {'id': this_id,
                     'feat': this_feat,
                     'label': this_label,
                     'label_emb': this_label_emb}
        feature = tf.train.Features(feature=feat_dict)
        example = tf.train.Example(features=feature)

        writer.write(example.SerializeToString())

    writer.close()


def save_meta_info(set_name, data):
    file_name = os.path.join('./data', set_name, 'meta_info.npy')
    save_dict = {'s_cls': data['s_cls'],
                 'u_cls': data['u_cls'],
                 'cls_emb': data['cls_emb'],
                 's_cls_emb': data['s_cls_emb'],
                 'u_cls_emb': data['u_cls_emb']}
    np.save(file_name, save_dict)


def build_dataset(set_name, root_folder):
    training_dict, seen_dict, unseen_dict = process_mat(set_name, root_folder)

    convert_tfrecord(training_dict, set_name, set_profiles.PART_NAME[0])
    convert_tfrecord(seen_dict, set_name, set_profiles.PART_NAME[1])
    convert_tfrecord(unseen_dict, set_name, set_profiles.PART_NAME[2])

    save_meta_info(set_name, training_dict)


if __name__ == '__main__':
    _root_folder = '/home/ymcidence/Workspace/MatlabWorkspace/xlsa17/data/'
    _set_name = 'SUN'
    build_dataset(_set_name, _root_folder)
    print('hehe')
