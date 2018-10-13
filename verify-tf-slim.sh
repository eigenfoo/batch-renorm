#!/bin/bash

# Tests if installation of tf.contrib.slim (via Tensorflow 1.0) is working
echo 'Verifying Tensorflow Slim...'
python -c "import tensorflow.contrib.slim as slim; eval = slim.evaluation.evaluate_once"
echo 'Success.\n'

# Tests if submodule of tensorflow/models/research/slim is present
echo 'Verifying Tensorflow Slim image models library...'
cd ./models/research/slim
python -c "from nets import cifarnet; mynet = cifarnet.cifarnet"
echo 'Success.\n'

echo 'Tensorflow Slim installations present and working.'
