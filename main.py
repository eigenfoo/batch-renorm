from inception_v3 import InceptionV3
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets.cifar10 import load_data
from tqdm import tqdm

# As specified in paper
MICROBATCH_SIZE = 32
NUM_MICROBATCHES = 50
BATCH_SIZE = MICROBATCH_SIZE * NUM_MICROBATCHES
NUM_EPOCHS = 30

NUM_CLASSES = 10
HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3

# Load data and split into train and val sets
(x_train, y_train), (x_val, y_val) = load_data()

# FIXME this is an ugly hack to make sure training data has a multiple of 1600
# of examples, for microbatching to work out.
x_train = x_train[:49600]
y_train = np.squeeze(y_train[:49600])

# FIXME ugly hack to avoid evaluating val in batches
x_val = x_val[:2000]
y_val = np.squeeze(y_val[:2000])

# Normalize and reshape data and labels
x_train, x_val = \
    map(lambda x: (x / 255.0).reshape([-1, HEIGHT, WIDTH, NUM_CHANNELS]),
        [x_train, x_val])
#y_train, y_val = \
#    map(lambda y: keras.utils.to_categorical(y, NUM_CLASSES),
#        [y_train, y_val])

x_train_batches = np.split(x_train, x_train.shape[0] // BATCH_SIZE)
y_train_batches = np.split(y_train, y_train.shape[0] // BATCH_SIZE)

images = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, NUM_CHANNELS])
labels = tf.placeholder(tf.int32, shape=[None])
training = tf.placeholder(bool, shape=[])

# Make model
predictions, loss, train_step, accuracy = InceptionV3(
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
sess.run(tf.local_variables_initializer())
# train_writer = tf.summary.FileWriter('./train_inspecc', sess.graph)

# Training
for i in range(NUM_EPOCHS):
    print('Epoch #{}: '.format(i))
    for x_batch, y_batch in tqdm(zip(x_train_batches, y_train_batches)):
        sess.run(train_step, feed_dict={images: x_batch,
                                        labels: y_batch,
                                        training: True})
    loss_, acc_ = sess.run([loss, accuracy],
                           feed_dict={images: x_batch,
                                      labels: y_batch,
                                      training: True})

    print('Train loss: {} - Train accuracy: {}'.format(loss_, acc_))

    # Validation
    loss_, acc_ = sess.run([loss, accuracy],
                           feed_dict={images: x_val,
                                      labels: y_val,
                                      training: False})

    print('Validation loss: {} - Validation accuracy: {}'.format(loss_, acc_))
