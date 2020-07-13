import math
import cv2
import json
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from config import *


def load_raw(filename):

    return cv2.imread(filename)[..., :: -1] / 255.


def load_RGB(filename):

    rgb = cv2.imread(filename)[..., :: -1] / 255.
    noise = np.random.normal(0, CONFIG_SAMPLES_STD_NOISE, rgb.shape)
    noisy = np.clip(rgb + noise, 0.0, 1.0)
    return noisy.astype(np.float32)


def load_grayscale(filename):

    gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) / 255.
    noise = np.random.normal(0, CONFIG_SAMPLES_STD_NOISE, gray.shape)
    noisy = np.clip(gray + noise, 0.0, 1.0)
    noisy = np.stack(3 * [noisy], axis=-1)
    return noisy.astype(np.float32)


class Dataset:

    def __init__(self, samples_dir, mode='train', shuffle_every_epoch=True, representation='rgb'):

        assert mode in ['train', 'val']
        assert representation in ['rgb', 'grayscale', 'raw']

        lines = open('{}/{}.txt'.format(samples_dir, mode)).readlines()
        info = json.load(open('{}/info.json'.format(samples_dir), 'r'))
        num_negatives = info['negatives_{}'.format(mode)]
        num_positives = info['positives_{}'.format(mode)]
        num_samples_per_class = min(num_positives, num_negatives)

        self.num_samples = 2 * num_samples_per_class
        self.curr_epoch = 1
        self.num_epochs = CONFIG_TRAIN_NUM_EPOCHS
        self.curr_batch = 1
        self.batch_size = CONFIG_TRAIN_BATCH_SIZE
        self.num_batches = math.ceil(self.num_samples / self.batch_size)
        self.shuffle_every_epoch = shuffle_every_epoch

        assert self.num_samples > self.batch_size

        # load data
        count = {'0': 0, '1': 0}
        labels = []
        filenames = []
        for line in lines:
            filename, label = line.split()
            if count[label] < num_samples_per_class:
                filenames.append(filename)
                labels.append(label)
                count[label] += 1

        self.sample_size = cv2.imread(filenames[0]).shape

        # data in array format
        # self.images = np.array(images).astype(np.float32)
        self.filenames = filenames
        self.labels = np.array(labels).astype(np.int32)
        if mode == 'train': self.labels = self._one_hot(self.labels)

        self.pool = multiprocessing.Pool(CONFIG_TRAIN_NUM_PROC)
        self.load_func = None
        if representation == 'rgb':
            self.load_func = load_RGB
        elif representation == 'grayscale':
            self.load_func = load_grayscale
        else:
            self.load_func = load_raw


    def __del__(self):

        self.pool.close()


    def _one_hot(self, labels):

        one_hot = np.zeros((labels.shape[0], CONFIG_NUM_CLASSES))
        one_hot[np.arange(labels.shape[0]), labels] = 1
        return one_hot


    def next_batch(self):

        assert self.curr_epoch <= self.num_epochs

        # shuffle dataset
        if (self.curr_batch == 1) and (self.shuffle_every_epoch):
            self.filenames, self.labels = shuffle(self.filenames, self.labels)

        # crop batch
        i1 = (self.curr_batch - 1) * self.batch_size
        i2 = i1 + self.batch_size
        filenames = self.filenames[i1 : i2]
        labels = self.labels[i1 : i2]

        images = self.pool.map(self.load_func, filenames)
        images = np.array(images)
        # # images must be in RGB (channels last) float32 format
        # images = np.stack([images, images, images], axis=0).transpose([1, 2, 3, 0]).astype(np.float32)

        # next epoch?
        if self.curr_batch == self.num_batches:
            self.curr_epoch += 1
            self.curr_batch = 1
        else:
            self.curr_batch += 1

        return images, labels