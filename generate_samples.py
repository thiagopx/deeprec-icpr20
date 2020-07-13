import os
import json
import shutil
import numpy as np
import cv2
import math
import random
import glob
import time
from skimage.filters import threshold_sauvola
from config import *

from docrec.strips.strips import Strips

# ISRI_DATASET_DIR = 'datasets/isri-ocr'

# # ignore images of the test set (dataset D2)
# ignore_images = ['9461_009.3B', '8509_001.3B', '8519_001.3B', '8520_003.3B', '9415_015.3B',
#                  '8541_001.3B', '8541_001.3B', '8541_001.3B', '8541_001.3B', '8541_001.3B',
#                  '9421_005.3B', '9421_005.3B', '9421_005.3B', '9462_056.3B', '8502_001.3B',
#                  '8518_001.3B', '8535_001.3B', '9413_018.3B', '8505_001.3B', '9462_096.3B']


def clear_dir(path):

    for root, _, files in os.walk(path):
        for file in files:
            os.remove('{}/{}'.format(root, file))


def generate_files(samples_dir):
    ''' Generate train.txt and val.txt. '''

    docs_neg_train = glob.glob('{}/negatives/train/*.jpg'.format(samples_dir))
    docs_neg_val = glob.glob('{}/negatives/val/*.jpg'.format(samples_dir))
    docs_pos_train = glob.glob('{}/positives/train/*.jpg'.format(samples_dir))
    docs_pos_val = glob.glob('{}/positives/val/*.jpg'.format(samples_dir))

    neg_train = ['{} 0'.format(doc) for doc in docs_neg_train]
    pos_train = ['{} 1'.format(doc) for doc in docs_pos_train]
    neg_val = ['{} 0'.format(doc) for doc in docs_neg_val]
    pos_val = ['{} 1'.format(doc) for doc in docs_pos_val]

    train = neg_train + pos_train
    val = neg_val + pos_val
    random.shuffle(train)
    random.shuffle(val)

    # save
    open('{}/train.txt'.format(samples_dir), 'w').write('\n'.join(train))
    open('{}/val.txt'.format(samples_dir), 'w').write('\n'.join(val))

    # general info
    info = {
        'negatives_train': len(neg_train),
        'positives_train': len(pos_train),
        'negatives_val': len(neg_val),
        'positives_val': len(pos_val)
    }
    json.dump(info, open('{}/info.json'.format(samples_dir), 'w'))



def apply_noise(sample, half, radius):
    ''' Apply noise onto a sample. '''

    noisy = sample.copy()
    noise = np.random.choice([False, True], (sample.shape[0], 2 * radius))
    noisy[:, half - radius : half + radius] |= noise
    return noisy


def generate_samples_pre_train(samples_dir, dataset):
    ''' Sampling process. '''

    assert dataset in ['cdip', 'isri-ocr']

    # create directory structure
    os.makedirs('{}/positives/train'.format(samples_dir), exist_ok=True)
    os.makedirs('{}/positives/val'.format(samples_dir), exist_ok=True)
    os.makedirs('{}/negatives/train'.format(samples_dir), exist_ok=True)
    os.makedirs('{}/negatives/val'.format(samples_dir), exist_ok=True)

    clear_dir(samples_dir)

    wr = CONFIG_SAMPLES_SIZE // 2 # rightmost pixels from the left strip
    wl = CONFIG_SAMPLES_SIZE - wr # leftmost pixels from the right strip

    neutral = lambda sample: ((~sample).sum() / sample.size) < CONFIG_SAMPLES_THRESH_BLACK
    noisy = lambda sample: apply_noise(sample, wr, CONFIG_SAMPLES_DISP_NOISE)

    # load training documents
    docs = []
    if dataset == 'cdip':
        docs = glob.glob('{}/ORIG_*/*.tif'.format(CONFIG_CDIP_DATASET_DIR))
    else:
        docs = glob.glob('{}/**/*.tif'.format(CONFIG_ISRI_DATASET_DIR), recursive=True)
    random.shuffle(docs)

    # split train and val sets
    num_docs = len(docs)
    docs_train = docs[int(CONFIG_SAMPLES_RATIO_VAL* num_docs) :]
    docs_val = docs[ : int(CONFIG_SAMPLES_RATIO_VAL * num_docs)]

    processed = 0
    for mode, docs in zip(['train', 'val'], [docs_train, docs_val]):
        count = {'positives': 0, 'negatives': 0}
        for doc in docs:

            print('gen. samples: processing document {}/{}[mode={}]\r'.format(processed + 1, num_docs, mode), end='')
            processed += 1

            image = cv2.imread(doc, cv2.IMREAD_GRAYSCALE)
            height, width = image.shape

            # threshold
            thresh = threshold_sauvola(image)
            thresholded = (image > thresh)
            acc = 0
            strips = []
            for i in range(CONFIG_SAMPLES_NUM_STRIPS):
                dw = int((width - acc) / (CONFIG_SAMPLES_NUM_STRIPS - i))
                strip = thresholded[:, acc : acc + dw]
                strips.append(strip)
                acc += dw

            # sampling
            pos_combs = [(i, i + 1) for i in range(CONFIG_SAMPLES_NUM_STRIPS - 1)]
            neg_combs = [(i, j) for i in range(CONFIG_SAMPLES_NUM_STRIPS) for j in range(CONFIG_SAMPLES_NUM_STRIPS) if (i != j) and (i + 1 != j)]
            random.shuffle(pos_combs)
            random.shuffle(neg_combs)

            # alternate sampling (pos1 -> neg1, pos2 -> neg2, ...)
            samples = dict(positives=[], negatives=[])
            for pos_comb, neg_comb in zip(pos_combs, neg_combs):
                for (i, j), label in zip([pos_comb, neg_comb], ['positives', 'negatives']):
                    image = np.hstack([strips[i][:, -wr :], strips[j][:, : wl]])
                    max_samples = CONFIG_SAMPLES_MAX_POS if label == 'positives' else CONFIG_SAMPLES_RATIO_NEG * CONFIG_SAMPLES_MAX_POS
                    for y in range(0, height - CONFIG_SAMPLES_SIZE, CONFIG_SAMPLES_STRIDE):
                        crop = image[y : y + CONFIG_SAMPLES_SIZE]
                        if not neutral(crop) and len(samples[label]) < max_samples:
                            sample = noisy(crop)
                            sample = (255 * sample).astype(np.uint8)
                            filename = '{}/{}/{}/{}.jpg'.format(samples_dir, label, mode, count[label])
                            samples[label].append((filename, sample))
                            count[label] += 1

            # # balancing
            # min_size = min(len(samples['positives']), len(samples['negatives']))
            # samples['positives'] = samples['positives'][: min_size]
            # samples['negatives'] = samples['negatives'][: min_size]

            # saving
            for filename, sample in samples['positives'] + samples['negatives']:
                cv2.imwrite(filename, sample)

    generate_files(samples_dir)


def generate_samples_fine_tuning(samples_dir, doc):
    ''' Sampling process. '''

    # create directory structure
    os.makedirs('{}/positives/train'.format(samples_dir), exist_ok=True)
    os.makedirs('{}/positives/val'.format(samples_dir), exist_ok=True)
    os.makedirs('{}/negatives/train'.format(samples_dir), exist_ok=True)
    os.makedirs('{}/negatives/val'.format(samples_dir), exist_ok=True)

    clear_dir(samples_dir)

    strips = Strips(path=doc, filter_blanks=True)
    num_strips = len(strips.strips)

    # positives (+)
    num_positives = 0
    positives = dict()
    for i, strip in enumerate(strips.strips):
        print('gen. samples: assembling positives :: strip {}/{}\r'.format(i + 1, num_strips), end='')
        h, w, _ = strip.image.shape
        positives[i] = dict()
        for r in range(0, h - CONFIG_SAMPLES_SIZE, CONFIG_SAMPLES_STRIDE_FT):
            for c in range(0, w - CONFIG_SAMPLES_SIZE, CONFIG_SAMPLES_STRIDE_FT):
                crop_mask = strip.mask[r : r + CONFIG_SAMPLES_SIZE, c: c + CONFIG_SAMPLES_SIZE]
                crop = strip.image[r : r + CONFIG_SAMPLES_SIZE, c : c + CONFIG_SAMPLES_SIZE]
                if (crop_mask.min() == 255): # crop fits inside the mask
                    try:
                        positives[i][r].append(crop.copy())
                    except KeyError:
                        positives[i][r] = [crop.copy()]
                    num_positives += 1
    print()

    # GRAYSCALE
    # for i, strip in enumerate(strips.strips):
    #     print('gen. samples: assembling positives :: strip {}/{}\r'.format(i + 1, len(strips.strips)), end='')
    #     h, w, _ = strip.image.shape
    #     # image = cv2.cvtColor(strip.image, cv2.COLOR_RGB2GRAY)
    #     import matplotlib.pyplot as plt
    #     plt.imshow(strip.image)
    #     plt.show()
    #     for i in range(0, h - CONFIG_SAMPLES_SIZE, CONFIG_SAMPLES_STRIDE_FT):
    #         for j in range(0, w - CONFIG_SAMPLES_SIZE, CONFIG_SAMPLES_STRIDE_FT):
    #             crop_mask = strip.mask[i : i + CONFIG_SAMPLES_SIZE, j: j + CONFIG_SAMPLES_SIZE]
    #             crop = image[i : i + CONFIG_SAMPLES_SIZE, j: j + CONFIG_SAMPLES_SIZE]
    #             if (crop_mask.min() == 255): # crop fits inside the mask
    #             # if (crop_mask.min() == 255) and ((crop != 255).sum() / crop.size >= CONFIG_SAMPLES_THRESH_BLACK): # crop fits inside the mask
    #                 positives.append(crop.copy())
    # print()


    # negatives (-)
    wl = math.ceil(CONFIG_SAMPLES_SIZE / 2)
    idx_strips = list(range(num_strips))
    negatives = []
    while len(negatives) < num_positives:
        print('gen. samples: assembling negatives :: sample {}/{}\r'.format(len(negatives) + 1, num_positives), end='')
        i = random.choice(idx_strips)
        r = random.choice(list(positives[i].keys()))
        j = random.choice([k for k in range(num_strips) if k != i])
        while r not in positives[j]:
            i = random.choice(idx_strips)
            r = random.choice(list(positives[i].keys()))
            j = random.choice([k for k in range(num_strips) if k != i])

        # print(i, j, r)
        sample_i = random.choice(positives[i][r])
        sample_j = random.choice(positives[j][r])

        left = sample_i[:, : wl]
        right = sample_j[:, wl :]
        negative = np.hstack([left, right])
        negatives.append(negative)
    print()

    # # noise generation
    # samples = positives + negatives
    # for i, sample in enumerate(samples):
    #     print('gen. samples: adding noise :: sample {}/{}\r'.format(i + 1, len(samples)), end='')
    #     start = (CONFIG_SAMPLES_SIZE - CONFIG_SAMPLES_DISP_NOISE) // 2
    #     f = np.random.randint(0, 255, (CONFIG_SAMPLES_SIZE, CONFIG_SAMPLES_DISP_NOISE)).astype(np.uint8)
    #     noise = np.random.randint(0, 255, (CONFIG_SAMPLES_SIZE, CONFIG_SAMPLES_DISP_NOISE)).astype(np.uint8)
    #     # for j in range(3): # for each channel
    #     #     sample[:, start : start + CONFIG_SAMPLES_DISP_NOISE, j] = cv2.add(sample[:, start : start + CONFIG_SAMPLES_DISP_NOISE, j], uniform)
    #     sample[:, start : start + CONFIG_SAMPLES_DISP_NOISE] = cv2.add(sample[:, start : start + CONFIG_SAMPLES_DISP_NOISE], uniform)
    #     # noise = np.random.randint(0, 255, (CONFIG_SAMPLES_SIZE, CONFIG_SAMPLES_DISP_NOISE)).astype(np.uint8)
    # print()

    # save to disk
    positives = [sample for samples_i in positives.values() for samples_r in samples_i.values() for sample in samples_r]
    print(len(positives), len(negatives))
    random.shuffle(positives)
    random.shuffle(negatives)

    num_positives = len(positives)
    num_negatives = len(negatives)
    positives_train = positives[int(CONFIG_SAMPLES_RATIO_VAL* num_positives) :]
    positives_val = positives[: int(CONFIG_SAMPLES_RATIO_VAL* num_positives)]
    negatives_train = negatives[int(CONFIG_SAMPLES_RATIO_VAL* num_negatives) :]
    negatives_val = negatives[: int(CONFIG_SAMPLES_RATIO_VAL* num_negatives)]

    for i, sample in enumerate(positives_train): cv2.imwrite('{}/positives/train/{}.jpg'.format(samples_dir, i), sample)
    for i, sample in enumerate(positives_val): cv2.imwrite('{}/positives/val/{}.jpg'.format(samples_dir, i), sample)
    for i, sample in enumerate(negatives_train): cv2.imwrite('{}/negatives/train/{}.jpg'.format(samples_dir, i), sample)
    for i, sample in enumerate(negatives_val): cv2.imwrite('{}/negatives/val/{}.jpg'.format(samples_dir, i), sample)

    generate_files(samples_dir)





