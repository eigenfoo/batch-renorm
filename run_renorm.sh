#!/bin/bash

source activate curro
python -u train_renorm.py 1
python -u train_renorm.py 2
python -u train_renorm.py 3
python -u train_renorm.py 4
python -u train_renorm.py 5
