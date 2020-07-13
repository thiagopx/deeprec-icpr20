import os
import shutil
import time
import random
import numpy as np
import tensorflow as tf

from config import *
from train import train_and_validate
from generate_samples import generate_samples_pre_train

from docrec.models.squeezenet import SqueezeNet


# seed experiment for reproducibility
random.seed(CONFIG_SEED)
np.random.seed(CONFIG_SEED)
tf.set_random_seed(CONFIG_SEED)

# repeat pre-training
t0 = time.time()
for dataset in ['cdip', 'isri-ocr']:
    for run in range(1, CONFIG_NUM_RUNS + 1):
        print('pre-train: dataset={} run={}'.format(dataset, run))

        # 1) samples generation
        samples_dir = '{}/pretrain/{}/{}'.format(CONFIG_SAMPLES_BASE_DIR, dataset, run)
        samples_dir = os.path.expanduser(samples_dir)
        generate_samples_pre_train(samples_dir, dataset)

        print()

        # 2) training and validation
        train_dir = '{}/pretrain/{}/{}'.format(CONFIG_TRAIN_BASE_DIR, dataset, run)
        train_dir = os.path.expanduser(train_dir)
        train_and_validate(samples_dir, train_dir, pretrained='imagenet', representation=CONFIG_TRAIN_REPRESENTATION)

elapsed = time.time() - t0

option = input('Do you want to remove the samples pretrain directory in order to save disk space? [y]/n: ').lower()
while option not in ['y', 'n']:
    option = input('Please, type y or n: ').lower()

if option == 'y':
    pretrain_dir = os.path.expanduser('{}/pretrain'.format(CONFIG_SAMPLES_BASE_DIR))
    shutil.rmtree(pretrain_dir)

print('Elapsed time: {:.2f} min.'.format(elapsed / 60.))