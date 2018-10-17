from foo import InceptionV3
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets.cifar10 import load_data

# As specified in paper
MICROBATCH_SIZE = 32
NUM_MICROBATCHES = 50
BATCH_SIZE = MICROBATCH_SIZE * NUM_MICROBATCHES
NUM_EPOCHS = 1

NUM_CLASSES = 10
HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3

# Load data and split into train, val, test sets
(x_train, y_train), (x_test, y_test) = load_data()
(x_val, y_val), (x_test, y_test) = \
            (x_test[:5000], y_test[:5000]), (x_test[5000:], y_test[5000:])

# FIXME this is an ugly hack to make sure all data has a multiple of 1600 of
# examples...
x_train = x_train[:49600]
y_train = y_train[:49600]
x_val = x_val[:4800]
y_val = y_val[:4800]
x_test = x_test[:4800]
y_test = y_test[:4800]

# Normalize and reshape data and labels
x_train, x_val, x_test = \
    map(lambda x: (x / 255.0).reshape([-1, HEIGHT, WIDTH, NUM_CHANNELS]),
        [x_train, x_val, x_test])
y_train, y_val, y_test = \
    map(lambda y: keras.utils.to_categorical(y, NUM_CLASSES),
        [y_train, y_val, y_test])

x_train_batches = np.split(x_train, x_train.shape[0] // BATCH_SIZE)
y_train_batches = np.split(y_train, y_train.shape[0] // BATCH_SIZE)

images = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, NUM_CHANNELS])
labels = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])
training = tf.placeholder(bool, shape=[])

# Make model
loss, train_step, accuracy = InceptionV3(
    images,
    labels,
    training,
    classes=NUM_CLASSES,
    renorm=False,
    microbatch_size=MICROBATCH_SIZE,
    num_microbatches=NUM_MICROBATCHES
)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(NUM_EPOCHS):
    print('Epoch #{}: '.format(i))
    for x_batch, y_batch in zip(x_train_batches, y_train_batches):
        sess.run(train_step, feed_dict={images: x_batch,
                                        labels: y_batch,
                                        training: True})
    loss_, acc_ = sess.run([loss, accuracy],
                           feed_dict={x: data_batch, y: label_batch})
    print('Loss: {:0.3f} - Accuracy: {0.3f}'.format(loss_, acc_))
