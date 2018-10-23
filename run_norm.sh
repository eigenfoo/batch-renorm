#!/bin/bash

source activate curro
CUDA_VISIBLE_DEVICES=4 python -u train_norm.py 1
CUDA_VISIBLE_DEVICES=4 python -u train_norm.py 2
CUDA_VISIBLE_DEVICES=4 python -u train_norm.py 3
CUDA_VISIBLE_DEVICES=4 python -u train_norm.py 4
CUDA_VISIBLE_DEVICES=4 python -u train_norm.py 5
