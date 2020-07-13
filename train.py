import os
import sys
import json
import time
import math
import numpy as np
import tensorflow as tf
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt

from config import *
from dataset import Dataset

from docrec.models.squeezenet import SqueezeNet


def train_and_validate(samples_dir, train_dir, pretrained='imagenet', representation='rgb'):

    ''' Training stage. '''

    # 1) train -----
    t0 = time.time() # start cron

    print('loading training samples :: ', end='')
    sys.stdout.flush()
    dataset = Dataset(samples_dir, mode='train', representation=representation)
    H, W, C = dataset.sample_size
    print('num_samples={} sample_size={}x{}'.format(dataset.num_samples, H, W))

    # setup train data directory
    os.makedirs('{}/model'.format(train_dir), exist_ok=True)

    # clear default graph
    tf.reset_default_graph()

    # placeholders
    images_ph = tf.placeholder(tf.float32, name='images_ph', shape=(None, H, W, C)) # channels last
    labels_ph = tf.placeholder(tf.float32, name='labels_ph', shape=(None, CONFIG_NUM_CLASSES)) # one-hot enconding
    global_step_ph = tf.placeholder(tf.int32, name='global_step_ph', shape=()) # control dropout of the model

    # architecture definition
    model = SqueezeNet(images_ph, num_classes=CONFIG_NUM_CLASSES, mode='train', channels_first=False)
    logits_op = tf.reshape(model.output, [-1, CONFIG_NUM_CLASSES]) # #batches x #classes (squeeze height dimension)

    # loss function
    loss_op = tf.losses.softmax_cross_entropy(onehot_labels=labels_ph, logits=logits_op)

    # learning rate definition
    num_steps_per_epoch = math.ceil(dataset.num_samples / CONFIG_TRAIN_BATCH_SIZE)
    total_steps = CONFIG_TRAIN_NUM_EPOCHS * num_steps_per_epoch
    decay_steps = math.ceil(CONFIG_TRAIN_STEP_SIZE * total_steps)
    learning_rate_op = tf.train.exponential_decay(CONFIG_TRAIN_INIT_LEARNING_RATE, global_step_ph, decay_steps, 0.1, staircase=True)

    # optimizer (adam method) and training operation
    optimizer = tf.train.AdamOptimizer(learning_rate_op, name='adam')
    train_op = optimizer.minimize(loss_op)

    # session setup
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # init graph
    sess.run(tf.global_variables_initializer())
    model.set_session(sess)
    if pretrained == 'imagenet':
        model.load_pretrained_imagenet()
    elif pretrained is not None:
        best_epoch = json.load(open('{}/info.json'.format(pretrained), 'r'))['best_epoch']
        weights_path = '{}/model/{}.npy'.format(pretrained, best_epoch)
        model.load_weights(weights_path)

    # writer = tf.summary.FileWriter('/home/tpaixao/graph_train', sess.graph)

    # training loop
    loss_sample = []
    loss_avg_per_sample = []
    steps = []
    global_step = 1
    for epoch in range(1, CONFIG_TRAIN_NUM_EPOCHS + 1):
        for step in range(1, num_steps_per_epoch + 1):
            # batch data
            images, labels = dataset.next_batch()
            # train
            learning_rate, loss, x = sess.run([learning_rate_op, loss_op, train_op], feed_dict={images_ph: images, labels_ph: labels, global_step_ph: global_step})
            # show training status
            loss_sample.append(loss)
            if (step % 10 == 0) or (step == num_steps_per_epoch):
                loss_avg_per_sample.append(np.mean(loss_sample))
                steps.append(global_step)
                elapsed = time.time() - t0
                remaining = elapsed * (total_steps - global_step) / global_step
                print('train: [{:.2f}%] step={}/{} epoch={} loss={:.5f} :: {:.2f}/{:.2f} seconds lr={}\r'.format(
                    100 * global_step / total_steps, global_step, total_steps, epoch,
                    loss_avg_per_sample[-1], elapsed, remaining, learning_rate
                ), end='')
                loss_sample = []
            global_step += 1
        # save epoch model
        model.save_weights('{}/model/{}.npy'.format(train_dir, epoch))
    sess.close()
    t_train = time.time() - t0 # stop cron

    # save training loss curve
    plt.plot(steps, loss_avg_per_sample)
    plt.savefig('{}/loss.png'.format(train_dir))
    loss_plt_data = dict(loss=loss_avg_per_sample, steps=steps)
    np.save('{}/loss_plt_data.npy'.format(train_dir), np.array(loss_plt_data))

    # 2) validation -----
    t0 = time.time()

    print('loading validation samples :: ', end='')
    sys.stdout.flush()
    dataset = Dataset(samples_dir, mode='val', representation=representation)
    H, W, C = dataset.sample_size
    print('num_samples={} sample_size={}x{}'.format(dataset.num_samples, H, W))

    # clear default graph
    tf.reset_default_graph()

    # placeholders
    images_ph = tf.placeholder(tf.float32, name='images_ph', shape=(None, H, W, C)) # channels last

    # architecture deifnition
    model = SqueezeNet(images_ph, num_classes=CONFIG_NUM_CLASSES, mode='inference', channels_first=False)
    logits_op = tf.reshape(model.output, [-1, CONFIG_NUM_CLASSES]) # #batches x #classes (squeeze height dimension)

    # predictions
    predictions_op = tf.argmax(logits_op, 1)

    # setup a new session
    sess = tf.Session(config=config)
    model.set_session(sess)

    # validation loop
    num_steps_per_epoch = math.ceil(dataset.num_samples / CONFIG_TRAIN_BATCH_SIZE)
    best_epoch = 0
    best_accuracy = 0.0
    accuracy_per_epoch = []
    for epoch in range(1, CONFIG_TRAIN_NUM_EPOCHS + 1):
        # load epoch model
        model.load_weights('{}/model/{}.npy'.format(train_dir, epoch))
        # writer = tf.summary.FileWriter('/home/tpaixao/graph_val/{}'.format(epoch), sess.graph)
        total_correct = 0
        for step in range(1, num_steps_per_epoch + 1):
            # batch data
            images, labels = dataset.next_batch()
            batch_size = images.shape[0]
            predictions = sess.run(predictions_op, feed_dict={images_ph: images})
            num_correct = np.sum(predictions==labels)
            total_correct += num_correct
            if (step % 10 == 0) or (step == num_steps_per_epoch):
                print('val: step={} accuracy={:.2f}\r'.format(step, 100 * num_correct / batch_size), end='')
        # epoch average accuracy
        accuracy = 100.0 * total_correct / dataset.num_samples
        accuracy_per_epoch.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
    sess.close()
    t_val = time.time() - t0 # stop cron

    print('val: best epoch={} accuracy={:.2f}'.format(best_epoch, best_accuracy))
    plt.clf()
    epochs = list(range(1, CONFIG_TRAIN_NUM_EPOCHS + 1))
    plt.plot(epochs, accuracy_per_epoch)
    plt.savefig('{}/accuracy.png'.format(train_dir))
    accuracy_plt_data = dict(accuracy=accuracy_per_epoch, epochs=epochs)
    np.save('{}/accuracy_plt_data.npy'.format(train_dir), np.array(accuracy_plt_data))

    # dump training info
    info = {
        'time_train': t_train,
        'time_val': t_val,
        'best_epoch': best_epoch
    }
    json.dump(info, open('{}/info.json'.format(train_dir), 'w'))
    print('train time={:.2f} min. val time={:.2f} min.'.format(t_train / 60., t_val / 60.))