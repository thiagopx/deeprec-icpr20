import os
import time
import cv2
import numpy as np
import sys
import gc

import matplotlib.pyplot as plt

from docrec.strips.strips import Strips


DOWNSAMPLING_FACTOR = 3
WINDOW_H = 1000
SHIFT_H = 250

top = 0
docs = ['datasets/images/D{:03d}'.format(i) for i in range(1, 6)]
fig, ax = plt.subplots(nrows=1, ncols=1)
doc_idx = 0
strip_idx = 0
strips = Strips(path=docs[doc_idx])
image = strips.image(highlight=True, highlight_idx=strip_idx)


def save(strips, ext='jpg'):

    path = docs[doc_idx]
    strips_path = '{}/strips'.format(path)
    masks_path = '{}/masks'.format(path)
    os.makedirs(strips_path, exist_ok=True)
    os.makedirs(masks_path, exist_ok=True)
    doc_id = path.split('/')[-1]
    for i in range(len(strips.strips)):
        strip = strips(i).image
        mask = strips(i).mask
        filename = '{}/{}{:02d}.{}'.format(strips_path, doc_id, i + 1, ext)
        cv2.imwrite(filename, strip[..., :: -1]) # RGB => BGR
        filename = '{}/{}{:02d}.npy'.format(masks_path, doc_id, i + 1)
        np.save(filename, mask)


def update_plot():

    global strips, strip_idx, image, top
    print(top)
    image = strips.image(highlight=True, highlight_idx=strip_idx)
    crop = image[top : top + WINDOW_H : DOWNSAMPLING_FACTOR, :: DOWNSAMPLING_FACTOR]
    ax.imshow(crop)
    fig.canvas.draw()
    gc.collect()


def press(event):

    global doc_idx, strip_idx, strips, top

    if event.key == 'p':
        doc_idx = max(0, doc_idx - 1)
        strips = Strips(path=docs[doc_idx])
        strip_idx = 0
        update_plot()
        print(event.key)
        return

    if event.key == 'n':
        doc_idx = min(len(docs) - 1, doc_idx + 1)
        strips = Strips(path=docs[doc_idx])
        strip_idx = 0
        update_plot()
        print(event.key)
        return

    if event.key in ['z', 'ctrl+z']:
        jump = 1 if event.key == 'z' else 5
        strip_idx = (strip_idx - jump) % len(strips.strips)
        update_plot()
        print(event.key)
        return

    if event.key in ['x', 'ctrl+x']:
        jump = 1 if event.key == 'x' else 5
        strip_idx = (strip_idx + jump) % len(strips.strips)
        update_plot()
        print(event.key)
        return

    if event.key in ['left',  'ctrl+left']:
        jump = 1 if event.key == 'left' else 5
        new_strip_idx = (strip_idx - jump) % len(strips.strips)
        # if new_strip_idx > strip_idx + 1:
        #     new_strip_idx -= 1
        strips.strips.insert(new_strip_idx, strips.strips.pop(strip_idx))
        # aux = strips.strips[strip_idx]
        # strips.strips[strip_idx] = strips.strips[new_strip_idx]
        # strips.strips[new_strip_idx] = aux
        strip_idx = new_strip_idx
        update_plot()
        print(event.key)
        return

    if event.key in ['right',  'ctrl+right']:
        jump = 1 if event.key == 'right' else 5
        new_strip_idx = (strip_idx + jump) % len(strips.strips)
        # if new_strip_idx > strip_idx + 1:
        #     new_strip_idx -= 1
        strips.strips.insert(new_strip_idx, strips.strips.pop(strip_idx))
        # aux = strips.strips[strip_idx]
        # strips.strips[strip_idx] = strips.strips[new_strip_idx]
        # strips.strips[new_strip_idx] = aux
        strip_idx = new_strip_idx
        update_plot()
        print(event.key)
        return

    if event.key == 'w':
        save(strips, ext='jpg')
        print(event.key)
        return

    if event.key == 'down':
        top = int(min(top + SHIFT_H, image.shape[0] - WINDOW_H))
        update_plot()
        print(event.key)
        return

    if event.key == 'up':
        top = int(max(top - SHIFT_H, 0))
        update_plot()
        print(event.key)
        return

    if event.key == 'escape':
        sys.exit(0)
    print(event.key)

if __name__ == '__main__':

    crop = image[top : top + WINDOW_H : DOWNSAMPLING_FACTOR, :: DOWNSAMPLING_FACTOR]
    ax.imshow(crop)
    fig.canvas.mpl_connect('key_press_event', press)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.axis('off')
    plt.show()
