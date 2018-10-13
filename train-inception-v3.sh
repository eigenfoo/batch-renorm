#!/bin/bash

DATASET_DIR=./tmp/data/cifar10
TRAIN_DIR=./tmp/train_logs
python models/research/slim/train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=cifar10 \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=inception_v3
