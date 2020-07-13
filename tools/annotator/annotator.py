import os
import time
import json
import cv2
import numpy as np
import sys
import gc

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QT5Agg')

DOWNSAMPLE_FACTOR = 3
FONTSIZE = 14
KEY_SCENARIO_MAP = {
    '0': 'typewritten',
    '1': 'handwritten',
    '2': 'running-text',
    '3': 'table-text',
    '4': 'form-text',
    '5': 'small-fontsize-text',
    '6': 'low-density-text',
    '7': 'table-or-grid-or-form', # graphical object
    '8': 'skew', # general layout
    '9': 'noisy' # # general appearance
}
ANNOTATION_FILE = 'annotation.json'
docs = ['datasets/S-CDIP/scanned/D{:03}.jpg'.format(i) for i in range(1, 101)]
annotation = {doc: [] for doc in docs}
if os.path.exists(ANNOTATION_FILE): annotation = json.load(open(ANNOTATION_FILE, 'r'))
doc_idx = 0
image = cv2.imread(docs[doc_idx])[:: DOWNSAMPLE_FACTOR, :: DOWNSAMPLE_FACTOR, :: -1]
fig, ax = plt.subplots(nrows=1, ncols=1)


def update_plot():


    image = cv2.imread(docs[doc_idx])[:: DOWNSAMPLE_FACTOR, :: DOWNSAMPLE_FACTOR, :: -1]
    ax.clear()
    ax.imshow(image)
    for i in range(len(KEY_SCENARIO_MAP)):
        label = KEY_SCENARIO_MAP[str(i)]
        color = 'red' if label in annotation[docs[doc_idx]] else 'gray'
    # for i, label in enumerate(annotation[docs[doc_idx]]):
        ax.text(0, 3 * FONTSIZE * i, '{}: {}'.format(i, label), fontsize=FONTSIZE, color=color)
    ax.axis('off')
    ax.set_title('{}'.format(docs[doc_idx].split('/')[-1]))
    fig.canvas.draw()
    gc.collect()


def press(event):

    global doc_idx

    if event.key in KEY_SCENARIO_MAP:
        if KEY_SCENARIO_MAP[event.key] in annotation[docs[doc_idx]]:
            annotation[docs[doc_idx]].remove(KEY_SCENARIO_MAP[event.key])
        else:
            annotation[docs[doc_idx]].append(KEY_SCENARIO_MAP[event.key])
            annotation[docs[doc_idx]].sort()
        update_plot()
    elif event.key == 'left':
        doc_idx = (doc_idx - 1) % len(docs)
        update_plot()
    elif event.key == 'right':
        doc_idx = (doc_idx + 1) % len(docs)
        update_plot()
    elif event.key == 'w':
        json.dump(annotation, open(ANNOTATION_FILE, 'w'))
        print('write!')

if __name__ == '__main__':

    ax.imshow(image)
    for i in range(len(KEY_SCENARIO_MAP)):
        label = KEY_SCENARIO_MAP[str(i)]
        color = 'red' if label in annotation[docs[doc_idx]] else 'gray'
    # for i, label in enumerate(annotation[docs[doc_idx]]):
        ax.text(0, 3 * FONTSIZE * i, '{}: {}'.format(i, label), fontsize=FONTSIZE, color=color)
    ax.set_title('{}'.format(docs[doc_idx].split('/')[-1]))
    ax.axis('off')
    fig.canvas.mpl_connect('key_press_event', press)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
