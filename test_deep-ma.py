import os
import sys
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

t0 = time.time() # start cron

# reconstruction instances
docs1 = ['datasets/S-MARQUES/mechanical/D{:03}'.format(i) for i in range(1, 62) if i != 3]
docs2 = ['datasets/S-ISRI-OCR/mechanical/D{:03}'.format(i) for i in range(1, 21)]
docs3 = ['datasets/S-CDIP/mechanical/D{:03}'.format(i) for i in range(1, 100)]
images = ['datasets/images/mechanical/D{:03}'.format(i) for i in range(1, 6)]

# solver
solver = SolverConcorde(maximize=True, max_precision=2, seed=CONFIG_SEED)

# reconstruction instances
processed = 1
records = []
for doc in docs:
    print('[{:.2f}%] doc={} :: '.format(100 * processed / len(docs), doc), end='')
    sys.stdout.flush()
    processed += 1
    _, dataset, _, doc_id = doc.split('/')
    train_dir = '{}/finetuning/{}/{}/1'.format(CONFIG_TRAIN_BASE_DIR, dataset, doc_id)
    best_epoch = json.load(open('{}/info.json'.format(train_dir), 'r'))['best_epoch']
    weights_path = '{}/model/{}.npy'.format(train_dir, best_epoch)
    algorithm = Proposed(
        'squeezenet', weights_path, 10, (3000, 31), num_classes=CONFIG_NUM_CLASSES,
        verbose=False, representation=CONFIG_TEST_REPRESENTATION_FT
    )
    strips = Strips(path=doc, filter_blanks=True).shuffle()
    init_permutation = strips.permutation()
    compatibilities = algorithm(strips=strips).compatibilities
    print(compatibilities[:3,:3])
    solution = solver(instance=compatibilities).solution
    acc = accuracy(solution, init_permutation)
    print('acc={:.2f}%'.format(100 * acc))
    sys.stdout.flush()
    records.append(['deep-ma', doc, acc, init_permutation, solution, compatibilities.tolist()])

os.makedirs('results', exist_ok=True)
json.dump(records, open('results/deep-ma.json', 'w'))
print('Elapsed time: {:.2f} min.'.format((time.time() - t0) / 60.))