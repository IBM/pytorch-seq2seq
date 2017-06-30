# Introduction

This is a framework for sequence-to-sequence (seq2seq) models implemented in [PyTorch](http://pytorch.org).  The framework has modularized and extensible components for seq2seq models, training and inference, checkpoints, etc.  This is an alpha release. We appreciate any kind of feedback or contribution.

## Roadmap
Seq2seq is a fast evolving field with new techniques and architectures being published frequently.  The goal of this library is facilitating the development of such techniques and applications.  While constantly improving the quality of code and documentation, we will focus on the following items:

* Evaluation with benchmarks such as WMT machine translation, COCO image captioning, conversational models, etc;
* Provide more flexible model options, improving the usability of the library;
* Adding latest architectures such as the CNN based model proposed by [Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122) and the transformer model proposed by [Attention Is All You Need](https://arxiv.org/abs/1706.03762);
* Support features in the new versions of PyTorch.

## Installation
This package requires Python 2.7. We recommend creating a new virtual environment for this project (using virtualenv or conda).  

### Prerequisites

* Numpy: `pip install numpy` (Refer [here](https://github.com/numpy/numpy) for problem installing Numpy).
* PyTorch: Refer to [PyTorch website](http://pytorch.org/) to install the version w.r.t. your environment.

### Install from source
Currently we only support installation from source code using setuptools.  Checkout the source code and run the following commands:

    pip install -r requirements.txt
    python setup.py install

If you already had a version of PyTorch installed on your system, please verify that the active torch package is at least version 0.1.11.

## Get Started
### Prepare toy dataset

	# Run script to generate the reverse toy dataset
    # The generated data is stored in data/toy_reverse by default
	scripts/toy.sh

### Train and play
	TRAIN_PATH=data/toy_reverse/train/data.txt
	DEV_PATH=data/toy_reverse/dev/data.txt
	# Start training
    python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH

It will take about 3 minutes to train on CPU and less than 1 minute with a Tesla K80.  Once training is complete, you will be prompted to enter a new sequence to translate and the model will print out its prediction (use ctrl-C to terminate).  Try the example below!

    Input:  1 3 5 7 9
	Expected output: 9 7 5 3 1 EOS

### Checkpoints
Checkpoints are organized by experiments and timestamps as shown in the following file structure

    experiment_dir
	+-- input_vocab
	+-- output_vocab
	+-- checkpoints
	|  +-- YYYY_mm_dd_HH_MM_SS
	   |  +-- decoder
	   |  +-- encoder
	   |  +-- model_checkpoint

The sample script by default saves checkpoints in the `experiment` folder of the root directory.  Look at the usages of the sample code for more options, including resuming and loading from checkpoints.

## Troubleshoots and Contributing
If you have any questions, bug reports, and feature requests, please [open an issue](https://github.com/IBM/pytorch-seq2seq/issues/new) on Github.

We appreciate any kind of feedback or contribution.  Feel free to proceed with small issues like bug fixes, documentation improvement.  For major contributions and new features, please discuss with the collaborators in corresponding issues.  

### Development Environment
We setup the development environment using [Vagrant](https://www.vagrantup.com/).  Run `vagrant up` with our 'Vagrantfile' to get started.

### Code Style
We follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for code style.  Especially the style of docstrings is important to generate documentation.
