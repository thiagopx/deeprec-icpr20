import os
import shutil
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


# # repeat pre-training
for run in range(1, CONFIG_NUM_RUNS + 1):

    print('pre-train: run={}'.format(run))

    # 1) samples generation
    samples_dir = '{}/pretrain/{}'.format(CONFIG_SAMPLES_BASE_DIR, run)
    samples_dir = os.path.expanduser(samples_dir)
    generate_samples_pre_train(samples_dir)

    # 2) training and validation
    train_dir = '{}/pretrain/{}'.format(CONFIG_TRAIN_BASE_DIR, run)
    train_dir = os.path.expanduser(train_dir)
    train_and_validate(samples_dir, train_dir)

