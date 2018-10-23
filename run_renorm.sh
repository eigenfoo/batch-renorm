#!/bin/bash

source activate curro
CUDA_VISIBLE_DEVICES=2 python -u train_renorm.py 1
CUDA_VISIBLE_DEVICES=2 python -u train_renorm.py 2
CUDA_VISIBLE_DEVICES=2 python -u train_renorm.py 3
CUDA_VISIBLE_DEVICES=2 python -u train_renorm.py 4
CUDA_VISIBLE_DEVICES=2 python -u train_renorm.py 5
