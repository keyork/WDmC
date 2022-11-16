import os
import shutil
import numpy as np


def split_data(raw_path, target_dir):

    raw_data = np.load(raw_path)
    train_data = raw_data["train"]
    train_label = raw_data["label_train"]
    test_data = raw_data["test"]

    base_idx = np.argwhere(train_label.sum(axis=1) <= 1)
    double_idx = np.argwhere(train_label.sum(axis=1) == 2)
    multi_idx = np.argwhere(train_label.sum(axis=1) > 2)
    b_d_idx = np.argwhere(train_label.sum(axis=1) <= 2)

    base_data = np.squeeze(train_data[base_idx], axis=1)
    base_label = np.squeeze(train_label[base_idx], axis=1)

    double_data = np.squeeze(train_data[double_idx], axis=1)
    double_label = np.squeeze(train_label[double_idx], axis=1)

    multi_data = np.squeeze(train_data[multi_idx], axis=1)
    multi_label = np.squeeze(train_label[multi_idx], axis=1)

    b_d_data = np.squeeze(train_data[b_d_idx], axis=1)
    b_d_label = np.squeeze(train_label[b_d_idx], axis=1)

    if not os.path.exists(os.path.join(target_dir, "base.npz")):
        print("{} -> {}".format(raw_path, os.path.join(target_dir, "base.npz")))
        np.savez(
            os.path.join(target_dir, "base.npz"),
            train=base_data,
            label_train=base_label,
        )
    if not os.path.exists(os.path.join(target_dir, "double.npz")):
        print("{} -> {}".format(raw_path, os.path.join(target_dir, "double.npz")))
        np.savez(
            os.path.join(target_dir, "double.npz"),
            train=double_data,
            label_train=double_label,
        )
    if not os.path.exists(os.path.join(target_dir, "multi.npz")):
        print("{} -> {}".format(raw_path, os.path.join(target_dir, "multi.npz")))
        np.savez(
            os.path.join(target_dir, "multi.npz"),
            train=multi_data,
            label_train=multi_label,
        )
    if not os.path.exists(os.path.join(target_dir, "bd.npz")):
        print("{} -> {}".format(raw_path, os.path.join(target_dir, "bd.npz")))
        np.savez(
            os.path.join(target_dir, "bd.npz"), train=b_d_data, label_train=b_d_label
        )
    if not os.path.exists(os.path.join(target_dir, "full.npz")):
        print("{} -> {}".format(raw_path, os.path.join(target_dir, "full.npz")))
        shutil.copyfile(raw_path, os.path.join(target_dir, "full.npz"))


def get_test_set(raw_path, target_path, scale=0.8):

    raw_data = np.load(raw_path)
    train_data = raw_data["train"]
    train_label = raw_data["label_train"]
    test_data = raw_data["test"]
    data_num = len(train_label)
    train_num = int(data_num * scale)
    idx_list = np.arange(data_num)
    np.random.shuffle(idx_list)
    spt_train_data = train_data[idx_list[:train_num]]
    spt_train_label = train_label[idx_list[:train_num]]
    spt_test_data = train_data[idx_list[train_num:]]
    spt_test_label = train_label[idx_list[train_num:]]
    if not os.path.exists(target_path):
        print("{} -> {}".format(raw_path, target_path))
        np.savez(
            target_path,
            train=spt_train_data,
            label_train=spt_train_label,
            test=spt_test_data,
            label_test=spt_test_label,
        )
