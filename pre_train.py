import os
import json
import argparse
import time
import random
import numpy as np
import multiprocessing
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from generate_samples import generate_samples_pre_train
from train import train_and_validate
from docrec.models.squeezenet import SqueezeNet
from config import *


# parameters processing
parser = argparse.ArgumentParser(description='Pre-train')
parser.add_argument(
    '-r', '--run', action='store', dest='run', required=False, type=int,
    default=1, help='Run.'
)
args = parser.parse_args()
# seed experiment for reproducibility
random.seed(CONFIG_SEED)
np.random.seed(CONFIG_SEED)
tf.set_random_seed(CONFIG_SEED)


# random.seed(CONFIG_SEED + args.run)
# np.random.seed(CONFIG_SEED + args.run)
# tf.set_random_seed(CONFIG_SEED + args.run)

# train_manager = TrainManager(samples_size=(CONFIG_SAMPLES_SIZE, CONFIG_SAMPLES_SIZE), CONFIG_NUM_RUNS)

# # repeat pre-training
for run in range(1, CONFIG_NUM_RUNS + 1):

    print('pre-train: run={}'.format(run))

    # 1) samples generation
    samples_dir = '{}/pretrain/{}'.format(CONFIG_SAMPLES_BASE_DIR, run)
    samples_dir = os.path.expanduser(samples_dir)
    # generate_samples_pre_train(samples_dir)

    # 2) training and validation
    train_dir = '{}/pretrain/{}'.format(CONFIG_TRAIN_BASE_DIR, run)
    train_dir = os.path.expanduser(train_dir)
    train_and_validate(samples_dir, train_dir)


# print('pre-train: run={}'.format(args.run))

# # 1) samples generation
# samples_dir = '{}/pretrain/{}'.format(CONFIG_SAMPLES_BASE_DIR, args.run)
# samples_dir = os.path.expanduser(samples_dir)
# # generate_samples_pre_train(samples_dir)

# # 2) training and validation
# train_dir = '{}/pretrain/{}'.format(CONFIG_TRAIN_BASE_DIR, args.run)
# train_dir = os.path.expanduser(train_dir)
# train_and_validate(samples_dir, train_dir)

    # gc.collect()

    # print('{}\r'.format(samples_dir), end='')
    # print(run)

# # training stage
# time_train = train(args)

# # validation
# best_epoch, time_val = validate(args)

# # dump training info
# info = {
#     'time_train': time_train,
#     'time_val': time_val,
#     'best_epoch': best_epoch,
#     'params': args.__dict__
# }
# json.dump(info, open('traindata/{}/info.json'.format(args.arch), 'w'))
# print('train time={:.2f} min. val time={:.2f} min.'.format(time_train / 60., time_val / 60.))