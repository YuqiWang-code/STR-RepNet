import os
import random
import numpy as np
from PIL import Image

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from changedetection.datasets import imutils


def load_img(path):
    """Load an RGB image as a float32 numpy array (H, W, 3)."""
    with Image.open(path) as im:
        img = np.asarray(im.convert("RGB"), dtype=np.float32)
    return img


def load_label(path):
    """Load a change mask and binarize it: gray >= 128 -> 1 else 0 (int64)."""
    with Image.open(path) as im:
        gray = np.asarray(im.convert("L"), dtype=np.uint8)
    return (gray >= 128).astype(np.int64)


class ChangeDetectionDataset(Dataset):
    """Bi-temporal change detection dataset in A/B/label + list format.

    Expected on-disk layout (per the RSML-3 contract):
        <dataset_root>/{A,B,label}/<sample_name>
        <dataset_root>/list/{train,val,test}.txt
    where each list line is a bare sample name (with its original extension),
    A = T1 image, B = T2 image, label = binary change mask (gray >= 128).
    """

    def __init__(self, dataset_path, data_list, crop_size=256, type='train', temporal_swap_prob=0.0):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.type = type
        self.crop_size = crop_size
        self.temporal_swap_prob = temporal_swap_prob

    def _transforms(self, pre_img, post_img, label):
        if self.type == 'train':
            pre_img, post_img, label = imutils.random_crop_new(pre_img, post_img, label, self.crop_size)
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)
            # temporal swap: (A,B)->(B,A), label unchanged
            if self.temporal_swap_prob > 0 and random.random() < self.temporal_swap_prob:
                pre_img, post_img = post_img, pre_img

        # ImageNet normalization (official VMamba normalization) for A/B only.
        pre_img = imutils.normalize_img(pre_img)
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, label

    def __getitem__(self, index):
        name = self.data_list[index]
        pre_img = load_img(os.path.join(self.dataset_path, 'A', name))
        post_img = load_img(os.path.join(self.dataset_path, 'B', name))
        label = load_label(os.path.join(self.dataset_path, 'label', name))

        pre_img, post_img, label = self._transforms(pre_img, post_img, label)
        return pre_img, post_img, label, name

    def __len__(self):
        return len(self.data_list)


def read_list(list_path):
    with open(list_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def make_data_loader(dataset_path, list_path, batch_size, crop_size=256, type='train',
                     num_workers=8, shuffle=None, drop_last=False):
    data_list = read_list(list_path)
    if shuffle is None:
        shuffle = (type == 'train')
    dataset = ChangeDetectionDataset(dataset_path, data_list, crop_size, type)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=drop_last, pin_memory=True)
