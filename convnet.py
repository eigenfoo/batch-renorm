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

import tensorflow as tf
from tensorflow import layers


def batch_norm(x, training, r_max, d_max,
               momentum=0.99, microbatch_size=1, epsilon=0.0001):
    '''
    # Arguments
        x: Input tensor. The zeroth dimension must be the batch dimension
        training: bool or tf.Bool. Set to True at training time.
        r_max: float or tf.Tensor. Threshold value to clip r by.
        d_max: float or tf.Tensor. Threshold value to clip d by.
        momentum: float. Momentum of moving averages.
        microbatch_size: int. Size of microbatches.

    # Returns
        Output tensor.

    Dimensionality notes:
    OLD: For non-microbatched renorm, we take moments of N H W C tensors over
    N H W to obtain moments of shape C. We then compute R and D per channel
    (shape N), clip them, and compute Y outputs.

    For microbatched batch renorm with N microbatches of M elements, we
    reshape N*M*H*W*C into N*M*H*W*C when training. We then take moments,
    reducing over the M, H, W dimensions (i.e. axis=[1 2 3]) to obtain
    moments of shape N*1*1*1*C. `r` and `d` are computed per microbatch
    (shape N*1*1*1*C) and are broadcast over the M H W C dimensions.
    Beta, gamma, mu, sigma, and their updates are all of shape C.
    '''
    channels = x.shape[-1]

    with(tf.variable_scope(None, 'batch_norm')):
        beta = tf.get_variable("beta", [channels], tf.float32,
                               tf.zeros_initializer)
        gamma = tf.get_variable("gamma", [channels], tf.float32,
                                tf.ones_initializer)

        mu = tf.get_variable("mu", [channels], tf.float32,
                             tf.zeros_initializer, trainable=False)
        sigma = tf.get_variable("sigma", [channels], tf.float32,
                                tf.ones_initializer, trainable=True)

        mu_old = tf.get_variable("mu_old", [channels], tf.float32,
                                 initializer=tf.zeros_initializer,
                                 trainable=False)
        sigma_old = tf.get_variable("sigma_old", [channels], tf.float32,
                                    initializer=tf.ones_initializer,
                                    trainable=False)

        # Train branch
        def train_step(x, mu, sigma, alpha):
            with(tf.variable_scope('train_branch')):
                x_shape = tf.shape(x)
                x_shaped = tf.reshape(x, [x_shape[0] // microbatch_size,
                                          microbatch_size,
                                          x.shape[1],
                                          x.shape[2],
                                          x.shape[3]])  # [N M H W C]
                mu_b, sigma_sq_b = tf.nn.moments(x_shaped, [1, 2, 3],
                                                 keep_dims=True)  # [N 1 1 1 C]
                sigma_b = tf.sqrt(sigma_sq_b)  # [N 1 1 1 C]

                with(tf.variable_scope('save_ops')):
                    mu_asgn_old = tf.assign(mu_old, mu, name='mu_save')
                    sigma_asgn_old = tf.assign(sigma_old, sigma,
                                               name='sigma_save')

                with(tf.control_dependencies([mu_asgn_old, sigma_asgn_old])):
                    with(tf.variable_scope('compute_updates')):
                        mu_add = alpha * tf.reduce_mean((mu_b - mu_asgn_old),
                                                        [0, 1, 2, 3])  # [C]
                        sigma_add = \
                            alpha * tf.reduce_mean((sigma_b - sigma_asgn_old),
                                                   [0, 1, 2, 3])  # [C]

                    with(tf.variable_scope('update_ops')):
                        mu_asgn = tf.assign_add(mu, mu_add, name='mu_update')
                        sigma_asgn = tf.assign_add(sigma, sigma_add,
                                                   name='sigma_update')

                    with(tf.control_dependencies([mu_asgn, sigma_asgn])):
                        with(tf.variable_scope('r')):
                            r = tf.stop_gradient(tf.clip_by_value(
                                sigma_b / sigma_asgn_old,
                                1 / r_max, r_max))
                        with(tf.variable_scope('d')):
                            d = tf.stop_gradient(tf.clip_by_value(
                                (mu_b - mu_asgn_old) / sigma_asgn_old,
                                -d_max, d_max))
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
              rmax=2,
              dmax=1,
              padding='same',
              strides=(1, 1),
              renorm=False,
              microbatch_size=32,
              num_microbatches=50,
              name=None):
    """Utility function to apply conv_2d and batch (re)norm.

    # Arguments
        x: input tensor.
        filters: filters in `conv2d`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `conv2d`.
        strides: strides in `conv2d`.
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

    bn_axis = -1  # Assume channels_last
    x = layers.conv2d(
        x,
        filters,
        (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)

    if not renorm:
        # Batch norm is simply with batch renorm with r_max = 1, d_max = 0
        x = batch_norm(x, training, 1, 0, microbatch_size=microbatch_size)
    else:
        x = batch_norm(x, training, rmax, dmax,
                       microbatch_size=microbatch_size)

    x = tf.nn.relu(x, name=name)
    return x


def make_conv_net(images,
                  labels,
                  training,
                  rmax,
                  dmax,
                  classes=10,
                  renorm=False,
                  microbatch_size=32,
                  num_microbatches=50,
                  **kwargs):
    """Instantiates a convolutional neural network.

    # Arguments
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        renorm: if True, apply batch renorm instead of batch norm.
        microbatch_size: number of examples in one microbatch.
        num_microbatches: number of microbatches in one minibatch.
    """
    channel_axis = -1  # Assume channels_last
    last = images

    last = conv2d_bn(last, 64, 3, 3, training, renorm=renorm,
                     rmax=rmax, dmax=dmax,
                     microbatch_size=microbatch_size,
                     num_microbatches=num_microbatches)
    last = conv2d_bn(last, 128, 3, 3, training, strides=(2, 2),
                     rmax=rmax, dmax=dmax,
                     renorm=renorm,
                     microbatch_size=microbatch_size,
                     num_microbatches=num_microbatches)
    last = tf.layers.dropout(last, 0.5, training=training)

    last = conv2d_bn(last, 256, 3, 3, training, strides=(2, 2),
                     rmax=rmax, dmax=dmax,
                     renorm=renorm,
                     microbatch_size=microbatch_size,
                     num_microbatches=num_microbatches)
    last = tf.layers.dropout(last, 0.5, training=training)

    last = conv2d_bn(last, 100, 1, 1, training, renorm=renorm,
                     rmax=rmax, dmax=dmax,
                     microbatch_size=microbatch_size,
                     num_microbatches=num_microbatches)

    last = tf.reduce_mean(last, axis=[1, 2])
    logits = tf.identity(last, name="logits")
    predictions = tf.argmax(logits, axis=1, output_type=tf.int32,
                            name='predictions')

    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    train_step = (tf.train.AdamOptimizer(learning_rate=0.0001)
                          .minimize(loss))

    correct_prediction = tf.equal(predictions, labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return predictions, loss, train_step, accuracy
