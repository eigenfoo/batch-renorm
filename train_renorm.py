'''
ECE471, Selected Topics in Machine Learning - Midterm Assignment
Suubmit by Oct. 24, 10pm.
tldr: Reproduce a subset of the results of a contemporary research paper.

Paper:
    Batch Renormalization: Towards Reducing Minibatch Dependence in
        Batch-Normalized Models, Sergey Ioffe
    http://ee.cooper.edu/~curro/cgml/week4/paper10.pdf

GitHub repository:
    https://github.com/eigenfoo/batch-renorm
'''

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets.cifar100 import load_data
from tqdm import tqdm
from convnet import make_conv_net

# As specified in paper
MICROBATCH_SIZE = 2
NUM_MICROBATCHES = 800
BATCH_SIZE = MICROBATCH_SIZE * NUM_MICROBATCHES
NUM_EPOCHS = 300

NUM_CLASSES = 100
HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3

# Load data and split into train and val sets
(x_train, y_train), (x_val, y_val) = load_data()

# FIXME this is an ugly hack to make sure training data has a multiple of 1600
# of examples, for microbatching to work out.
x_train = x_train[:49600]
y_train = np.squeeze(y_train[:49600])

y_val = np.squeeze(y_val)

# Normalize and reshape data and labels
x_train, x_val = \
    map(lambda x: (x / 255.0).reshape([-1, HEIGHT, WIDTH, NUM_CHANNELS]),
        [x_train, x_val])

x_train_batches = np.split(x_train, x_train.shape[0] // BATCH_SIZE)
y_train_batches = np.split(y_train, y_train.shape[0] // BATCH_SIZE)

images = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, NUM_CHANNELS])
labels = tf.placeholder(tf.int32, shape=[None])
training = tf.placeholder(bool, shape=[])

rmax = tf.placeholder(tf.float32, [])
dmax = tf.placeholder(tf.float32, [])

# Make model
predictions, loss, train_step, accuracy = make_conv_net(
    images,
    labels,
    training,
    rmax,
    dmax,
    classes=NUM_CLASSES,
    renorm=True,
    microbatch_size=MICROBATCH_SIZE,
    num_microbatches=NUM_MICROBATCHES
)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

accs = []


def get_rmax(num_epoch):
    thresh_epoch = 20
    if num_epoch < thresh_epoch:
        return 1
    else:
        return 1 + 0.5*(num_epoch - thresh_epoch)/(NUM_EPOCHS - thresh_epoch)


def get_dmax(num_epoch):
    thresh_epoch = 20
    if num_epoch < thresh_epoch:
        return 0
    else:
        return 0.5*(num_epoch - thresh_epoch)/(NUM_EPOCHS - thresh_epoch)


# Training
for i in range(NUM_EPOCHS):
    print('Epoch #{}: '.format(i))
    for x_batch, y_batch in tqdm(zip(x_train_batches, y_train_batches)):
        sess.run(train_step, feed_dict={images: x_batch,
                                        labels: y_batch,
                                        rmax: get_rmax(i),
                                        dmax: get_dmax(i),
                                        training: True})
    loss_, acc_ = sess.run([loss, accuracy],
                           feed_dict={images: x_batch,
                                      labels: y_batch,
                                      rmax: get_rmax(i),
                                      dmax: get_dmax(i),
                                      training: True})

    print('Train loss: {} - Train accuracy: {}'.format(loss_, acc_))

    # Validation
    tacc = 0
    for i in range(5):
        loss_, acc_ = sess.run([loss, accuracy],
                               feed_dict={images: x_val[i*2000:i*2000+2000],
                                          labels: y_val[i*2000:i*2000+2000],
                                          rmax: get_rmax(i),  # Ignored since
                                          dmax: get_dmax(i),  # training=False
                                          training: False})
        tacc = tacc + (acc_/5.0)
    accs.append(tacc)

    print('Validation loss: {} - Validation accuracy: {}'.format(loss_, tacc))

df = pd.DataFrame(data=accs,
                  columns=['Validation Accuracy'])
df.index = 31*df.index
df.to_csv('val_accs_renorm_{}.csv'.format(sys.argv[1]))
