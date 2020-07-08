import math
import cv2
import numpy as np
from multiprocessing import Pool

from sklearn.utils import shuffle

from config import *


def load_image(filename):

    return cv2.imread(filename)[..., :: -1]


class Dataset:

    def __init__(self, samples_dir, mode='train', shuffle_every_epoch=True):

        assert mode in ['train', 'val']

        lines = open('{}/{}.txt'.format(samples_dir, mode)).readlines()
        # info = json.load(open('{}/info.json'.format(samples_dir), 'r'))
        # num_negatives = info['stats']['negatives_{}'.format(mode)]
        # num_positives = info['stats']['positives_{}'.format(mode)]

        self.num_samples = len(lines)
        self.curr_epoch = 1
        self.num_epochs = CONFIG_TRAIN_NUM_EPOCHS
        self.curr_batch = 1
        self.batch_size = CONFIG_TRAIN_BATCH_SIZE
        self.num_batches = math.ceil(self.num_samples / self.batch_size)
        self.shuffle_every_epoch = shuffle_every_epoch

        assert self.num_samples > self.batch_size

        # load data
        # count = {'0': 0, '1': 0}
        labels = []
        filenames = []
        for line in lines:
            filename, label = line.split()
            filenames.append(filename)
            labels.append(label)

        self.sample_size = cv2.imread(filenames[0]).shape

        # data in array format
        # self.images = np.array(images).astype(np.float32)
        self.filenames = filenames
        self.labels = np.array(labels).astype(np.int32)
        if mode == 'train':
            self.labels = self._one_hot(self.labels)

        self.pool = multiprocessing.Pool(CONFIG_TRAIN_NUM_PROC)


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

        images = self.pool.map(load_image, filenames)
        # images = [cv2.imread(filename)[..., :: -1] for filename in filenames]
        images = np.array(images).astype(np.float32) / 255.

        # # images must be in RGB (channels last) float32 format
        # images = np.stack([images, images, images], axis=0).transpose([1, 2, 3, 0]).astype(np.float32)

        # next epoch?
        if self.curr_batch == self.num_batches:
            self.curr_epoch += 1
            self.curr_batch = 1
        else:
            self.curr_batch += 1

        return images, labels