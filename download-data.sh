#!/bin/bash

DATA_DIR=./tmp/data/cifar10
python models/research/slim/download_and_convert_data.py \
    --dataset_name=cifar10 \
    --dataset_dir=${DATA_DIR}

# Now we can easily create a TF-Slim dataset descriptor...
# https://github.com/tensorflow/models/tree/master/research/slim#creating-a-tf-slim-dataset-descriptor
