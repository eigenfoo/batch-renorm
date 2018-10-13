# Batch Renormalization

A Tensorflow implementation of batch renormalization, first introduced by Sergey
Ioffe in [this paper](https://arxiv.org/abs/1702.03275) (**arXiv:1702.03275v2
[cs.LG]**).

The goal of this project is to reproduce the following figure from the paper:

![Figure 2 from paper](https://raw.githubusercontent.com/eigenfoo/batch-renorm/master/docs/paper-figure.png)

## Installation, Setup and Verification

Assuming you have Tensorflow 1.0 installed, no more setup should be required.
Running `sh verify-tf-slim.sh` should return no errors.  If not, follow [these
instructions](https://github.com/tensorflow/models/tree/master/research/slim#installation)
to install both Tensorflow-Slim and its image models library.

## Download Data

Running `sh download-data.sh` should download and process the CIFAR-10 dataset.
This may take several minutes.
