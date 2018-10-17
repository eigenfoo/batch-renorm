import tensorflow as tf
from tensorflow import layers


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              training,
              padding='same',
              strides=(1, 1),
              renorm=False,
              microbatch_size=32,
              num_microbatches=50,
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        renorm: if True, apply batch renorm instead of batch norm.
        microbatch_size: number of examples in one microbatch.
        num_microbatches: number of microbatches in one minibatch.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `conv_2d` and `batch_normalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    bn_axis = 3  # Assume channels_last
    x = layers.conv2d(
        x,
        filters,
        (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)

    if not renorm:
        # Technically, microbatch batch normalization calls for updating
        # gradient's moments sequentially from microbatch batch to microbatch
        # batch. However, the Tensorflow implementation does _not_ do this,
        # instead averaging over the moments. This is exactly what we need. See
        # https://github.com/tensorflow/tensorflow/blob/c19e29306ce1777456b2dbb3a14f511edf7883a8/tensorflow/python/keras/layers/normalization.py#L581-L585
        x = layers.batch_normalization(x, axis=bn_axis, scale=False,
                                       virtual_batch_size=microbatch_size,
                                       training=training, name=bn_name)
    else:
        # TODO(andrey) implement batch renorm with a tf.keras.layers interface
        pass

    x = tf.nn.relu(x, name=name)
    return x


def InceptionV3(images,
                labels,
                training,
                classes=10,
                renorm=False,
                microbatch_size=32,
                num_microbatches=50,
                **kwargs):
    """Instantiates the Inception v3 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        renorm: if True, apply batch renorm instead of batch norm.
        microbatch_size: number of examples in one microbatch.
        num_microbatches: number of microbatches in one minibatch.
    """
    channel_axis = 3  # Assume channels_last

    x = conv2d_bn(images, 32, 3, 3, training, strides=(2, 2),
                  padding='valid', renorm=renorm,
                  microbatch_size=microbatch_size,
                  num_microbatches=num_microbatches)
    x = conv2d_bn(x, 32, 3, 3, training, padding='valid',
                  renorm=renorm, microbatch_size=microbatch_size,
                  num_microbatches=num_microbatches)
    x = conv2d_bn(x, 64, 3, 3, training,
                  renorm=renorm, microbatch_size=microbatch_size,
                  num_microbatches=num_microbatches)
    x = layers.max_pooling2d(x, (3, 3), strides=(2, 2))

    x = conv2d_bn(x, 80, 1, 1, training, padding='valid',
                  renorm=renorm, microbatch_size=microbatch_size,
                  num_microbatches=num_microbatches)
    x = conv2d_bn(x, 192, 3, 3, training, padding='valid',
                  renorm=renorm, microbatch_size=microbatch_size,
                  num_microbatches=num_microbatches)
    x = layers.max_pooling2d(x, (3, 3), strides=(2, 2))

    # mixed 0: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1, training, renorm=renorm,
                          microbatch_size=microbatch_size,
                          num_microbatches=num_microbatches)

    branch5x5 = conv2d_bn(x, 48, 1, 1, training, renorm=renorm,
                          microbatch_size=microbatch_size,
                          num_microbatches=num_microbatches)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, training, renorm=renorm,
                          microbatch_size=microbatch_size,
                          num_microbatches=num_microbatches)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, training, renorm=renorm,
                             microbatch_size=microbatch_size,
                             num_microbatches=num_microbatches)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, training, renorm=renorm,
                             microbatch_size=microbatch_size,
                             num_microbatches=num_microbatches)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, training, renorm=renorm,
                             microbatch_size=microbatch_size,
                             num_microbatches=num_microbatches)

    branch_pool = layers.average_pooling2d(x, (3, 3),
                                           strides=(1, 1),
                                           padding='same')
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1, training, renorm=renorm,
                            microbatch_size=microbatch_size,
                            num_microbatches=num_microbatches)
    x = tf.concat(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1, training, renorm=renorm,
                          microbatch_size=microbatch_size,
                          num_microbatches=num_microbatches)

    branch5x5 = conv2d_bn(x, 48, 1, 1, training, renorm=renorm,
                          microbatch_size=microbatch_size,
                          num_microbatches=num_microbatches)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, training, renorm=renorm,
                          microbatch_size=microbatch_size,
                          num_microbatches=num_microbatches)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, training, renorm=renorm,
                             microbatch_size=microbatch_size,
                             num_microbatches=num_microbatches)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, training, renorm=renorm,
                             microbatch_size=microbatch_size,
                             num_microbatches=num_microbatches)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, training, renorm=renorm,
                             microbatch_size=microbatch_size,
                             num_microbatches=num_microbatches)

    branch_pool = layers.average_pooling2d(x, (3, 3),
                                           strides=(1, 1),
                                           padding='same')
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1, training, renorm=renorm,
                            microbatch_size=microbatch_size,
                            num_microbatches=num_microbatches)
    x = tf.concat(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1, training, renorm=renorm,
                          microbatch_size=microbatch_size,
                          num_microbatches=num_microbatches)

    branch5x5 = conv2d_bn(x, 48, 1, 1, training, renorm=renorm,
                          microbatch_size=microbatch_size,
                          num_microbatches=num_microbatches)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, training, renorm=renorm,
                          microbatch_size=microbatch_size,
                          num_microbatches=num_microbatches)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, training, renorm=renorm,
                             microbatch_size=microbatch_size,
                             num_microbatches=num_microbatches)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, training, renorm=renorm,
                             microbatch_size=microbatch_size,
                             num_microbatches=num_microbatches)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, training, renorm=renorm,
                             microbatch_size=microbatch_size,
                             num_microbatches=num_microbatches)

    branch_pool = layers.average_pooling2d(x, (3, 3),
                                           strides=(1, 1),
                                           padding='same')
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1, training, renorm=renorm,
                            microbatch_size=microbatch_size,
                            num_microbatches=num_microbatches)
    x = tf.concat(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # Classification block
    # Global average pooling. Assumes channels_last
    x = tf.reduce_mean(x, axis=[1, 2])
    logits = layers.dense(x, classes, activation=None, name='logits')
    predictions = tf.nn.softmax(logits, name='predictions')

    loss = tf.losses.softmax_cross_entropy(labels, logits)
    train_step = (tf.train.RMSPropOptimizer(learning_rate=0.001)
                          .minimize(loss))

    correct_prediction = tf.equal(tf.round(predictions), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return predictions, loss, train_step, accuracy
