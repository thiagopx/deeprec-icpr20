# Elapsed time: 1347.58 min
import os
import sys
import shutil
import time
import random
import numpy as np
import tensorflow as tf

from config import *
from train import train_and_validate
from generate_samples import generate_samples_fine_tuning, generate_files, clear_dir

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

# training with isri-ocr
group1 = docs1 + docs3 + images
# training with cdip
group2 = docs2

# repeat fine-tuning
t0 = time.time()
processed = 1
total = len(group1 + group2)
records = []
for docs, pretrain_dataset in zip([group1, group2], ['isri-ocr', 'cdip']):
    for doc in docs:
        samples_dir_run1 = '{}/finetuning/1'.format(CONFIG_SAMPLES_BASE_DIR)
        samples_dir_run1 = os.path.expanduser(samples_dir_run1)
        for run in range(1, CONFIG_NUM_RUNS + 1):
            print('fine-tuning: pretrain dataset={} doc={} run={}'.format(pretrain_dataset, doc, run))
            sys.stdout.flush()

            # 1) samples generation
            # Obs.: it will overwritten the previous document samples
            samples_dir = '{}/finetuning/{}'.format(CONFIG_SAMPLES_BASE_DIR, run)
            samples_dir = os.path.expanduser(samples_dir)
            if run == 1: # generate images for run 1
                generate_samples_fine_tuning(samples_dir, doc)
            else: # generate new train/val splits and keep the images generated for the run 1
                os.makedirs(samples_dir, exist_ok=True)
                link_pos_src = '{}/positives'.format(samples_dir_run1)
                link_pos_tgt = '{}/positives'.format(samples_dir)
                link_neg_src = '{}/negatives'.format(samples_dir_run1)
                link_neg_tgt = '{}/negatives'.format(samples_dir)
                if os.path.exists(link_pos_tgt): os.unlink(link_pos_tgt)
                if os.path.exists(link_neg_tgt): os.unlink(link_neg_tgt)
                os.symlink(link_pos_src, link_pos_tgt, target_is_directory=True)
                os.symlink(link_neg_src, link_neg_tgt, target_is_directory=True)
                generate_files(samples_dir)

            # 2) training and validation
            pretrain_dir = '{}/pretrain/{}/{}'.format(CONFIG_TRAIN_BASE_DIR, pretrain_dataset, run)
            pretrain_dir = os.path.expanduser(pretrain_dir)
            _, dataset, _, doc_id = doc.split('/')
            train_dir = '{}/finetuning/{}/{}/{}'.format(CONFIG_TRAIN_BASE_DIR, dataset, doc_id, run)
            train_dir = os.path.expanduser(train_dir)
            train_and_validate(samples_dir, train_dir, pretrained=pretrain_dir, representation=CONFIG_TRAIN_REPRESENTATION_FT)

elapsed = time.time() - t0

option = input('Do you want to remove the samples finetuning directory in order to save disk space? [y]/n: ').lower()
while option not in ['y', 'n']:
    option = input('Please, type y or n: ').lower()
if option == 'y':
    finetuning_dir = os.path.expanduser('{}/finetuning'.format(CONFIG_SAMPLES_BASE_DIR))
    shutil.rmtree(finetuning_dir)

print('Elapsed time: {:.2f} min.'.format(elapsed / 60.))