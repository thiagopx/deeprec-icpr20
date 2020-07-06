import gc
import os
import json
import time
import math
import cv2
import random
import numpy as np
import multiprocessing
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from docrec.models.squeezenet import SqueezeNet
from config import *

NUM_CLASSES = 2

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
images_ph = tf.placeholder(tf.float32, name='images_ph', shape=(None, 31, 31, 3)) # channels last
model = SqueezeNet(images_ph, num_classes=1000, mode='inference', channels_first=False)
sess.run(tf.global_variables_initializer())
model.set_session(sess)
model.load_pretrained_imagenet()
# tf.train.Saver()

        # print('F')
        # # print('trainable==================')
        # # tvars = tf.trainable_variables()
        # # tvars_vals = sess.run(tvars)
        # # for var, val in zip(tvars, tvars_vals):
        # #     print(var.name)

        # # print('global==================')
        # # tvars = tf.global_variables()
        # # tvars_vals = sess.run(tvars)
        # # print(len(tvars))
        # # for var, val in zip(tvars, tvars_vals):
        # #     print(var.name)
        #     # if 'Conv1/kernel:0' in var.name or 'Conv_Class/kernel:0' in var.name:
        #     #      print(var.name, val)
        #         # break
        # # import sys
        # # sys.exit(0)

        # # setup train data directory
        # os.makedirs('{}/model'.format(train_dir), exist_ok=True)

        # # training loop
        # loss_sample = []
        # loss_avg_per_sample = []
        # steps = []
        # global_step = 1
        # for epoch in range(1, CONFIG_TRAIN_NUM_EPOCHS + 1):
        #     for step in range(1, num_steps_per_epoch + 1):
        #         # batch data
        #         images, labels = sess.run(next_batch_op)
        #         # train
        #         learning_rate, loss, x = sess.run([learning_rate_op, loss_op, train_op], feed_dict={images_ph: images, labels_ph: labels, global_step_ph: global_step})
        #         # show training status
        #         loss_sample.append(loss)
        #         if (step % 10 == 0) or (step == num_steps_per_epoch):
        #             loss_avg_per_sample.append(np.mean(loss_sample))
        #             steps.append(global_step)
        #             elapsed = time.time() - t0
        #             remaining = elapsed * (total_steps - global_step) / global_step
        #             print('train: [{:.2f}%] step={}/{} epoch={} loss={:.5f} :: {:.2f}/{:.2f} seconds lr={}\r'.format(
        #                 100 * global_step / total_steps, global_step, total_steps, epoch,
        #                 loss_avg_per_sample[-1], elapsed, remaining, learning_rate
        #             ), end='')
        #             loss_sample = []
        #         global_step += 1
        #     # save epoch model
        #     model.save_weights('{}/model/{}.npy'.format(train_dir, epoch))
        # sess.close()
        # t_train = time.time() - t0 # stop cron

        # # save training loss curve
        # plt.plot(steps, loss_avg_per_sample)
        # plt.savefig('{}/loss.png'.format(train_dir))
        # loss_plt_data = dict(loss=loss_avg_per_sample, steps=steps)
        # np.save('{}/loss_plt_data.npy'.format(train_dir), np.array(loss_plt_data))
