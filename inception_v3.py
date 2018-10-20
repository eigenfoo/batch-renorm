import tensorflow as tf
from tensorflow import layers

from tensorflow.python import debug as tf_debug


def batch_norm(x, training, r_max, d_max, momentum=0.99, microbatch_size=1, epsilon=0.0001):
    '''
    :param x: Input tensor. The zeroth dimension must be the batch dimension
    :param training: A python boolean or a tf.bool tensor.
    :return: The output tensor.
    '''

    # Dimensionality notes:
    # OLD: For non-microbatched renorm, we take moments of N H W C tensors over N H W to obtain moments of shape C
    # We then compute R and D per channel (shape N), clip them, and compute Y outputs
    #
    #
    # For microbatched batch renorm with N microbatches of M elements, we reshape N*M H W C into N M H W C when training
    # We then take moments, reducing over the M H W dimensions (i.e. [1 2 3]) to obtain moments of shape N 1 1 1 C
    # R and D are computed per microbatch (shape N 1 1 1 C) and are broadcast over the M H W C dimensions
    # Beta, gamma, mu, sigma, and their updates are all of shape C.

    channels = x.shape[-1]

    with(tf.variable_scope(None, 'batch_norm')):
        beta = tf.get_variable("beta", [channels], tf.float32, tf.zeros_initializer)
        gamma = tf.get_variable("gamma", [channels], tf.float32, tf.ones_initializer)

        mu = tf.get_variable("mu", [channels], tf.float32, tf.zeros_initializer, trainable=False)
        sigma = tf.get_variable("sigma", [channels], tf.float32, tf.ones_initializer, trainable=True)

        mu_old = tf.get_variable("mu_old", [channels], tf.float32, trainable=False, initializer=tf.zeros_initializer)
        sigma_old = tf.get_variable("sigma_old", [channels], tf.float32, trainable=False,
                                    initializer=tf.ones_initializer)

        # Train branch
        def train_step(x, mu, sigma, alpha):
            with(tf.variable_scope('train_branch')):
                x_shape = tf.shape(x)
                x_shaped = tf.reshape(x, [x_shape[0] // microbatch_size, microbatch_size, x.shape[1], x.shape[2],
                                          x.shape[3]])  # [N M H W C]
                mu_b, sigma_sq_b = tf.nn.moments(x_shaped, [1, 2, 3], keep_dims=True)  # [N 1 1 1 C]
                sigma_b = tf.sqrt(sigma_sq_b)  # [N 1 1 1 C]

                with(tf.variable_scope('save_ops')):
                    mu_asgn_old = tf.assign(mu_old, mu, name='mu_save')
                    sigma_asgn_old = tf.assign(sigma_old, sigma, name='sigma_save')

                with(tf.control_dependencies([mu_asgn_old, sigma_asgn_old])):
                    with(tf.variable_scope('compute_updates')):
                        mu_add = alpha * tf.reduce_mean((mu_b - mu_asgn_old), [0, 1, 2, 3])  # [C]
                        sigma_add = alpha * tf.reduce_mean((sigma_b - sigma_asgn_old), [0, 1, 2, 3])  # [C]

                    with(tf.variable_scope('update_ops')):
                        mu_asgn = tf.assign_add(mu, mu_add, name='mu_update')
                        sigma_asgn = tf.assign_add(sigma, sigma_add, name='sigma_update')

                    with(tf.control_dependencies([mu_asgn, sigma_asgn])):
                        with(tf.variable_scope('r')):
                            r = tf.stop_gradient(tf.clip_by_value(sigma_b / sigma_asgn_old, 1 / r_max, r_max))
                        with(tf.variable_scope('d')):
                            d = tf.stop_gradient(tf.clip_by_value((mu_b - mu_asgn_old) / sigma_asgn_old, -d_max, d_max))
                        with(tf.variable_scope('y_train')):
                            x_hat = ((x_shaped - mu_b) / sigma_b) * r + d
                            y = gamma * x_hat + beta

            return tf.reshape(y, x_shape)

        result = tf.cond(training, lambda: train_step(x, mu, sigma, momentum),
                         lambda: gamma * ((x - mu) / sigma) + beta)
        return result


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

  
def batchnorm_debug():
    global sess, bn, inp, tr
    inp = tf.placeholder(tf.float32, [None, 1, 1, 2])
    tr = tf.placeholder(tf.bool, [])
    sess = tf.Session()
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    bn = batch_norm(inp, tr, 99999, 99999, 0.50, 2)
    train_writer = tf.summary.FileWriter('./BNzz', sess.graph)
    sess.run(tf.global_variables_initializer())
    print(sess.run(bn, {inp: [[[[1, 1]]], [[[2, 2]]], [[[1, 1]]], [[[2, 2]]]], tr: True}))
    print(sess.run(bn, {inp: [[[[1, 1]]], [[[2, 2]]], [[[1, 1]]], [[[2, 2]]]], tr: True}))


batchnorm_debug()
