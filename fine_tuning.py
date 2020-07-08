import os
import random
import numpy as np
import tensorflow as tf

from config import *
# from train import train_and_validate
from generate_samples import generate_samples_fine_tuning

from docrec.models.squeezenet import SqueezeNet


# seed experiment for reproducibility
random.seed(CONFIG_SEED)
np.random.seed(CONFIG_SEED)
tf.set_random_seed(CONFIG_SEED)

# reconstruction instances
docs1 = ['datasets/S-MARQUES/mechanical/D{:03}'.format(i) for i in range(1, 62) if i != 3]
docs2 = ['datasets/S-ISRI-OCR/mechanical/D{:03}'.format(i) for i in range(1, 21)]
docs3 = ['datasets/S-CDIP/mechanical/D{:03}'.format(i) for i in range(1, 100)]
images = ['datasets/images/mechanical/D{:03}'.format(i) for i in range(1, 6)]

docs = docs1 + docs2 + docs3 + images
docs = [docs[-1]]

# repeat fine-tuning
for doc in docs:
    for run in range(1, CONFIG_NUM_RUNS + 1):

        print('fine-tuning: run={}'.format(run))

        # 1) samples generation
        # Obs.: it will overwritten the previous document samples
        samples_dir = '{}/finetuning/{}'.format(CONFIG_SAMPLES_BASE_DIR, run)
        samples_dir = os.path.expanduser(samples_dir)
        generate_samples_fine_tuning(samples_dir, doc)
        print(samples_dir)

        # 2) training and validation
        train_dir = '{}/finetuning/{}'.format(CONFIG_TRAIN_BASE_DIR, run)
        train_dir = os.path.expanduser(train_dir)
        train_and_validate(samples_dir, train_dir)