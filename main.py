from inception_v3 import InceptionV3
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets.cifar10 import load_data

# As specified in paper
GHOST_SIZE = 32
NUM_GHOSTS = 50
BATCH_SIZE = GHOST_SIZE * NUM_GHOSTS
NUM_EPOCHS = 50

NUM_CLASSES = 10
HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3


class AccuracyHistory(keras.callbacks.Callback):
    ''' Class for Keras callbacks. '''
    def on_train_begin(self, logs={}):
        self.acc = []
        self.topk_acc = []

    def on_batch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        self.topk_acc.append(logs.get('top_k_categorical_accuracy'))


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

# Instantiate, compile and train Inception-v3 model
model = InceptionV3(
    include_top=True,
    weights=None,  # Random initialization
    input_shape=[HEIGHT, WIDTH, NUM_CHANNELS],
    pooling='avg',  # Global average pooling on output of the last conv layer
    classes=NUM_CLASSES,
    renorm=False,
    ghost_size=GHOST_SIZE,
    num_ghosts=NUM_GHOSTS
)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy', 'top_k_categorical_accuracy'])

history = AccuracyHistory()
model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS,
          verbose=1,
          validation_data=[x_val, y_val],
          callbacks=[history])

# Save accuracy and top k accuracy.
pd.Series(data=history.acc).to_csv('acc.csv')
# pd.Series(data=history.topk_acc).to_csv('topk_acc.csv')

loss, acc, topk_acc = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', loss)
print('Test accuracy:', acc)
print('Test top k accuracy:', topk_acc)
