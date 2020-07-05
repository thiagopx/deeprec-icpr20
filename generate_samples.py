import os
import numpy as np
import cv2
import math
import random
import glob
import time

from config import *

ISRI_DATASET_DIR = 'datasets/isri-ocr'

# ignore images of the test set (dataset D2)
ignore_images = ['9461_009.3B', '8509_001.3B', '8519_001.3B', '8520_003.3B', '9415_015.3B',
                 '8541_001.3B', '8541_001.3B', '8541_001.3B', '8541_001.3B', '8541_001.3B',
                 '9421_005.3B', '9421_005.3B', '9421_005.3B', '9462_056.3B', '8502_001.3B',
                 '8518_001.3B', '8535_001.3B', '9413_018.3B', '8505_001.3B', '9462_096.3B']


def generate_samples_pre_train(samples_dir):
    ''' Sampling process. '''

    # create directory structure
    os.makedirs('{}/positives/train'.format(samples_dir), exist_ok=True)
    os.makedirs('{}/positives/val'.format(samples_dir), exist_ok=True)
    os.makedirs('{}/negatives/train'.format(samples_dir), exist_ok=True)
    os.makedirs('{}/negatives/val'.format(samples_dir), exist_ok=True)

    if glob.glob('{}/**/*.jpg'.format(samples_dir)):
        print('Sampling already done!')
        return

    docs = glob.glob('{}/**/*.tif'.format(ISRI_DATASET_DIR), recursive=True)

    # filter documents in ignore list
    docs = [doc for doc in docs if os.path.basename(doc).replace('.tif', '') not in ignore_images]
    random.shuffle(docs)

    # split train and val sets
    num_docs = len(docs)
    docs_train = docs[int(CONFIG_SAMPLES_RATIO_VAL* num_docs) :]
    docs_val = docs[ : int(CONFIG_SAMPLES_RATIO_VAL * num_docs)]

    processed = 0
    for mode, docs in zip(['train', 'val'], [docs_train, docs_val]):
        count = {'positives': 0, 'negatives': 0}
        for doc in docs:
            max_per_doc = 0

            print('gen. samples: processing document {}/{}[mode={}]\r'.format(processed + 1, num_docs, mode), end='')
            processed += 1

            # shredding
            # print('     => Shredding')
            image = cv2.imread(doc)
            h, w, c = image.shape
            acc = 0
            strips = []
            for i in range(CONFIG_SAMPLES_NUM_STRIPS):
                dw = int((w - acc) / (CONFIG_SAMPLES_NUM_STRIPS - i))
                strip = image[:, acc : acc + dw]
                noise_left = np.random.randint(0, 255, (h, CONFIG_SAMPLES_DISP_NOISE)).astype(np.uint8)
                noise_right = np.random.randint(0, 255, (h, CONFIG_SAMPLES_DISP_NOISE)).astype(np.uint8)
                for j in range(c): # for each channel
                    strip[:, : CONFIG_SAMPLES_DISP_NOISE, j] = cv2.add(strip[:, : CONFIG_SAMPLES_DISP_NOISE, j], noise_left)
                    strip[:, -CONFIG_SAMPLES_DISP_NOISE :, j] = cv2.add(strip[:, -CONFIG_SAMPLES_DISP_NOISE :, j], noise_right)
                strips.append(strip)
                acc += dw

            # positives
            wr = math.ceil(CONFIG_SAMPLES_SIZE / 2) # rightmost pixels from the left strip (strip i)
            wl = CONFIG_SAMPLES_SIZE - wr # leftmost pixels from the right strip (strip j)
            # print('     => Positive samples')
            N = len(strips)
            combs = [(i, i + 1) for i in range(N - 1)]
            random.shuffle(combs)
            stop = False
            for i, j in combs:
                # print('[{}][{}] :: total={}\r'.format(i, j, count['positives']), end='')
                image = np.hstack([strips[i][:, -wr :], strips[j][:, : wl]])
                for y in range(0, h - CONFIG_SAMPLES_SIZE, CONFIG_SAMPLES_STRIDE):
                    crop = image[y : y + CONFIG_SAMPLES_SIZE]
                    if (crop != 255).sum() / crop.size >= CONFIG_SAMPLES_THRESH_BLACK:
                        count['positives'] += 1
                        max_per_doc += 1
                        cv2.imwrite('{}/positives/{}/{}.jpg'.format(samples_dir, mode, count['positives']), crop)
                        if max_per_doc == CONFIG_SAMPLES_MAX_POS:
                            stop = True
                            break
                if stop:
                    break


            # print('     => Negative samples')
            # negatives
            combs = [(i, j) for i in range(N) for j in range(N) if (i != j) and (i + 1 != j)]
            random.shuffle(combs)
            stop = False
            for i, j in combs:
                # print('[{}][{}] :: total={}\r'.format(i, j, count['negatives']), end='')
                image = np.hstack([strips[i][:, -wr :], strips[j][:, : wl]])
                for y in range(0, h - CONFIG_SAMPLES_SIZE, CONFIG_SAMPLES_STRIDE):
                    crop = image[y : y + CONFIG_SAMPLES_SIZE]
                    if (crop != 255).sum() / crop.size >= CONFIG_SAMPLES_THRESH_BLACK:
                        count['negatives'] += 1
                        cv2.imwrite('{}/negatives/{}/{}.jpg'.format(samples_dir, mode, count['negatives']), crop)
                        if count['negatives'] >= int(CONFIG_SAMPLES_RATIO_NEG * count['positives']):
                            stop = True
                            break
                if stop:
                    break

    generate_txt(samples_dir)


def generate_samples_fine_tuning(samples_dir):
    ''' Sampling process. '''

    # create directory structure
    os.makedirs('{}/positives/train'.format(samples_dir), exist_ok=True)
    os.makedirs('{}/positives/val'.format(samples_dir), exist_ok=True)
    os.makedirs('{}/negatives/train'.format(samples_dir), exist_ok=True)
    os.makedirs('{}/negatives/val'.format(samples_dir), exist_ok=True)

    if glob.glob('{}/**/*.jpg'.format(samples_dir)):
        print('Sampling already done!')
        return

    # reconstruction instances
    docs1 = ['datasets/D1/mechanical/D{:03}'.format(i) for i in range(1, 62) if i != 3]
    docs2 = ['datasets/D2/mechanical/D{:03}'.format(i) for i in range(1, 21)]
    docs3 = ['datasets/D3/mechanical/D{:03}'.format(i) for i in range(1, 41)]
    docs = docs1 + docs2 + docs3

    # filter documents in ignore list
    docs = [doc for doc in docs if os.path.basename(doc).replace('.tif', '') not in ignore_images]
    random.shuffle(docs)

    # split train and val sets
    num_docs = len(docs)
    docs_train = docs[int(CONFIG_SAMPLES_RATIO_VAL* num_docs) :]
    docs_val = docs[ : int(CONFIG_SAMPLES_RATIO_VAL * num_docs)]

    processed = 0
    for mode, docs in zip(['train', 'val'], [docs_train, docs_val]):
        count = {'positives': 0, 'negatives': 0}
        for doc in docs:
            max_per_doc = 0

            print('Processing document {}/{}[mode={}]\r'.format(processed + 1, num_docs, mode), end='')
            processed += 1

            # shredding
            # print('     => Shredding')
            image = cv2.imread(doc)
            h, w, c = image.shape
            acc = 0
            strips = []
            for i in range(CONFIG_SAMPLES_NUM_STRIPS):
                dw = int((w - acc) / (CONFIG_SAMPLES_NUM_STRIPS - i))
                strip = image[:, acc : acc + dw]
                noise_left = np.random.randint(0, 255, (h, CONFIG_SAMPLES_DISP_NOISE)).astype(np.uint8)
                noise_right = np.random.randint(0, 255, (h, CONFIG_SAMPLES_DISP_NOISE)).astype(np.uint8)
                for j in range(c): # for each channel
                    strip[:, : CONFIG_SAMPLES_DISP_NOISE, j] = cv2.add(strip[:, : CONFIG_SAMPLES_DISP_NOISE, j], noise_left)
                    strip[:, -CONFIG_SAMPLES_DISP_NOISE :, j] = cv2.add(strip[:, -CONFIG_SAMPLES_DISP_NOISE :, j], noise_right)
                strips.append(strip)
                acc += dw

            # positives
            wr = math.ceil(CONFIG_SAMPLES_SIZE / 2) # rightmost pixels from the left strip (strip i)
            wl = CONFIG_SAMPLES_SIZE - wr # leftmost pixels from the right strip (strip j)
            # print('     => Positive samples')
            N = len(strips)
            combs = [(i, i + 1) for i in range(N - 1)]
            random.shuffle(combs)
            stop = False
            for i, j in combs:
                # print('[{}][{}] :: total={}\r'.format(i, j, count['positives']), end='')
                image = np.hstack([strips[i][:, -wr :], strips[j][:, : wl]])
                for y in range(0, h - CONFIG_SAMPLES_SIZE, CONFIG_SAMPLES_STRIDE):
                    crop = image[y : y + CONFIG_SAMPLES_SIZE]
                    if (crop != 255).sum() / crop.size >= CONFIG_SAMPLES_THRESH_BLACK:
                        count['positives'] += 1
                        max_per_doc += 1
                        cv2.imwrite('{}/positives/{}/{}.jpg'.format(samples_dir, mode, count['positives']), crop)
                        if max_per_doc == CONFIG_SAMPLES_MAX_POS:
                            stop = True
                            break
                if stop:
                    break


            # print('     => Negative samples')
            # negatives
            combs = [(i, j) for i in range(N) for j in range(N) if (i != j) and (i + 1 != j)]
            random.shuffle(combs)
            stop = False
            for i, j in combs:
                # print('[{}][{}] :: total={}\r'.format(i, j, count['negatives']), end='')
                image = np.hstack([strips[i][:, -wr :], strips[j][:, : wl]])
                for y in range(0, h - CONFIG_SAMPLES_SIZE, CONFIG_SAMPLES_STRIDE):
                    crop = image[y : y + CONFIG_SAMPLES_SIZE]
                    if (crop != 255).sum() / crop.size >= CONFIG_SAMPLES_THRESH_BLACK:
                        count['negatives'] += 1
                        cv2.imwrite('{}/negatives/{}/{}.jpg'.format(samples_dir, mode, count['negatives']), crop)
                        if count['negatives'] >= int(CONFIG_SAMPLES_RATIO_NEG * count['positives']):
                            stop = True
                            break
                if stop:
                    break

    generate_txt(samples_dir)



def generate_txt(samples_dir):
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