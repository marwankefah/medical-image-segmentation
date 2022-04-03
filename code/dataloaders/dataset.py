import os
import torch
import random
import numpy as np
from scipy.ndimage.interpolation import zoom
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import pandas as pd
from torch.utils.data import Dataset


class BaseFetaDataSets(Dataset):

    def __init__(self, configs=None, split='train', transform=None, teacher_transform=None):
        self._base_dir = configs.root_path
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.teacher_transform = teacher_transform
        self.configs = configs
        self.labeled_idxs = []
        self.unlabeled_idxs = []

        try:
            df = pd.read_csv(self._base_dir + '/data.csv', delimiter=",")
        except Exception as e:
            print("Error: please try and close the csv file, or check whether path is correct", e)
            raise

            # optimize by sending
        if self.split == 'train':
            dfTrainSubset = df[(df['datamode'] == 'train_labelled') | (df['datamode'] == 'train_unlabelled')]
            dfTrainSubset = dfTrainSubset.reset_index()
            self.sample_list = list(
                zip(dfTrainSubset['image'].values.tolist(), dfTrainSubset['manual'].values.tolist()))

            self.labeled_idxs = dfTrainSubset.index[dfTrainSubset['datamode'] == 'train_labelled'].tolist()
            self.unlabeled_idxs = dfTrainSubset.index[dfTrainSubset['datamode'] == 'train_unlabelled'].tolist()

        elif self.split == 'val':
            dfTrainSubset = df[df['datamode'] == 'val_labelled']
            dfTrainSubset = dfTrainSubset.reset_index()

            self.sample_list = list(
                zip(dfTrainSubset['image'].values.tolist(), dfTrainSubset['manual'].values.tolist()))
            self.labeled_idxs = dfTrainSubset.index[dfTrainSubset['datamode'] == 'val_labelled'].tolist()

        elif self.split == 'test':
            dfTrainSubset = df[df['datamode'] == 'test_labelled']
            dfTrainSubset = dfTrainSubset.reset_index()
            self.sample_list = list(
                zip(dfTrainSubset['image'].values.tolist(), dfTrainSubset['manual'].values.tolist()))
            self.labeled_idxs = dfTrainSubset.index[dfTrainSubset['datamode'] == 'test_labelled'].tolist()
        elif self.split == 'train_labelled':
            dfTrainSubset = df[(df['datamode'] == 'train_labelled')]
            dfTrainSubset = dfTrainSubset.reset_index()

            self.sample_list = list(
                zip(dfTrainSubset['image'].values.tolist(), dfTrainSubset['manual'].values.tolist()))
            # TODO check the indicies is actually true
            self.labeled_idxs = dfTrainSubset.index[dfTrainSubset['datamode'] == 'train_labelled'].tolist()
        elif self.split == 'train_unlabelled':
            dfTrainSubset = df[(df['datamode'] == 'train_unlabelled')]
            dfTrainSubset = dfTrainSubset.reset_index()

            self.sample_list = list(
                zip(dfTrainSubset['image'].values.tolist(), dfTrainSubset['manual'].values.tolist()))
            # TODO check the indicies is actually true
            self.unlabeled_idxs = dfTrainSubset.index[dfTrainSubset['datamode'] == 'train_unlabelled'].tolist()

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        img_path = os.path.join(self.configs.img_root_path, self.sample_list[idx][0][1:])
        labels_path = os.path.join(self.configs.img_root_path, self.sample_list[idx][1][1:])

        # if the data is labeled
        if self.sample_list[idx][1] != 'none':

            if self.transform:
                transformed = self.transform({"image": img_path, "label": labels_path})
            else:
                print("no transforms for training labeled?")
                raise
        else:
            #unlabeled data input to the teacher
            if self.teacher_transform:
                transformed = self.teacher_transform({"image": img_path})
                #TODO modify when RGB is input
                transformed['label'] = torch.zeros_like(transformed['image'])
            else:
                print("no transforms for training unlabeled?")
                raise

        sample = {'image': transformed['image'], 'label': transformed['label'], "idx": idx}

        return sample

    def get_labels(self):
        return self.y_test


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)

        label = torch.from_numpy(label.astype(np.uint8))

        sample = {'image': image, 'label': label}
        return sample


class ResizeTransform(object):
    def __init__(self, output_size, mode='train_unlabeled'):
        self.output_size = output_size
        self.mode = mode

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
