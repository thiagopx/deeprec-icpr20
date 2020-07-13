import os
import time
import json
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from config import *

from docrec.metrics import accuracy
from docrec.strips.strips import Strips
from docrec.compatibility.proposed import Proposed
from docrec.solver.solverconcorde import SolverConcorde


# seed experiment for reproducibility
random.seed(CONFIG_SEED)
np.random.seed(CONFIG_SEED)
tf.set_random_seed(CONFIG_SEED)

# parameters processing
parser = argparse.ArgumentParser(description='Test Proposed')
parser.add_argument(
    '-m', '--method', action='store', dest='method', required=False, type=str,
    default='deep', help='Method for Network architecture [squeezenet or mobilenet].'
)
args = parser.parse_args()

assert args.method in ['deep', 'deep-ma']


def build_method(method, doc=None):

    # train data directory
    train_dir = '{}/pretrain/1'.format(CONFIG_TRAIN_BASE_DIR)
    if method == 'deep-ma':
        assert doc is not None
        train_dir = '{}/finetuning/{}/1'.format(CONFIG_TRAIN_BASE_DIR, doc)

    best_epoch = json.load(open('{}/info.json'.format(train_dir), 'r'))['best_epoch']
    weights_path = '{}/model/{}.npy'.format(train_dir, best_epoch)
    algorithm = Proposed(
        'squeezenet', weights_path, 10, (3000, 31), num_classes=2, verbose=False,
        apply_thresh=(args.method=='deep'), thresh='sauvola'
    )
    return algorithm


t0 = time.time() # start cron

# reconstruction instances
docs1 = ['datasets/S-MARQUES/mechanical/D{:03}'.format(i) for i in range(1, 62) if i != 3]
docs2 = ['datasets/S-ISRI-OCR/mechanical/D{:03}'.format(i) for i in range(1, 21)]
docs3 = ['datasets/S-CDIP/mechanical/D{:03}'.format(i) for i in range(1, 100)]
images = ['datasets/images/mechanical/D{:03}'.format(i) for i in range(1, 6)]
docs = docs1 + docs2 + docs3 + images

# reconstruction pipeline (deep)
train_dir = '{}/pretrain/1'.format(CONFIG_TRAIN_BASE_DIR)
best_epoch = json.load(open('{}/info.json'.format(train_dir), 'r'))['best_epoch']
weights_path = '{}/model/{}.npy'.format(train_dir, best_epoch)
deep = build_method('deep')

# reconstruction pipeline (compatibility algorithm + solver)
best_epoch = json.load(open('{}/info.json'.format(train_dir), 'r'))['best_epoch']
weights_path = '{}/model/{}.npy'.format(train_dir, best_epoch)
algorithm = Proposed('squeezenet', weights_path, 10, (3000, 31), num_classes=2, verbose=False, apply_thresh=(args.method=='deep'), thresh='sauvola')
solver = SolverConcorde(maximize=True, max_precision=2)

# reconstruction instances
processed = 1
records = []
for doc in docs:
    print('[{:.2f}%] method={} doc={} :: '.format(100 * processed / len(docs), args.method, doc), end='')
    processed += 1
    strips = Strips(path=doc, filter_blanks=True).shuffle()
    init_permutation = strips.permutation()
    compatibilities = algorithm(strips=strips).compatibilities
    solution = solver(instance=compatibilities).solution
    acc = accuracy(solution, init_permutation)
    print('acc={:.2f}%\r'.format(100 * acc), end='')
    records.append([args.method, doc, acc, init_permutation, solution, compatibilities.tolist()])

os.makedirs('results', exist_ok=True)
json.dump(records, open('results/{}.json'.format(args.method), 'w'))
print('Elapsed time: {:.2f} min.'.format((time.time() - t0) / 60.))