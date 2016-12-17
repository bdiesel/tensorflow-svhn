# Project 5: Deep Learning Capstone Project. Decoding a Sequence of Digits in Real World Photos.

This project is a [TensorFlow](https://www.tensorflow.org/) implementation of a 
[Convolution Nueral Network](https://en.wikipedia.org/wiki/Convolutional_neural_network)
to decode a sequence of digits in the realworld using trained on 
the [SVHN dataset](http://ufldl.stanford.edu/housenumbers/). This project is partial 
fulfillment of the Udacity Machine Learning nanodegree program. 

## Installation
```
git clone https://github.com/bdiesel/machine-learning.git
cd ./machine-learning/projects/capstone
```


## Libraries and Dependicies 
The following Python 2.7 libraries are required:
* h5py
* matplotlib
* numpy
* PIL
* tensorflow
* scipy
* six
* sklearn



##Step 1: Download and preprocess the SVHN data
Download and preprocess the SVHN data using the svhn_data.py utility.

`python svhn_data.py`

This should generate a data folder data\svhn with two sub-directories cropped and full

The cropped cropped directory should contain 2 newly downloaded .mat files amd 6 numpy file for each dataset which wil be used for training.

The full directory should contain two sub directories test and full which contain png images of various sizes. The full directory should also contain 6 numpy files and and 3 tar.gz files, the tar.gz files may be removed to free up disk space.


##Step 2: Train your own models
First train the classifer. The weights generated here will be resused in the training of the multi-digit reader.  Note this step may take some time depending of your hardware. Pre-trained models can be downloaded from here for the single digit reader [Single Digit Weights](https://s3.amazonaws.com/tensorflow-weights/classifier.ckpt), and here for the [Multi-Digit Weights](https://s3.amazonaws.com/tensorflow-weights/regression.ckpt).

`python train_classifier.py`

This should generate a tensorflow checkpoint file:

`classifier.ckpt`

Next train the multi-digit reader

`python train_regressor.py`

This should generate a tensorflow checkpoint file:

`regression.ckpt`


##Step 3: Check model performance
The training files generate logs which can be used with 
[TensorBoard](https://www.tensorflow.org/versions/r0.10/how_tos/summaries_and_tensorboard/index.html) to inspect the models performance overtime. 

To view the single digit performance after running train_classifier.py run:

`tensorboard --logdir="/tmp/svhn_classifier_logs"`

To view the multi digit performance after running train_regressor.py run:

`tensorboard --logdir="/tmp/svhn_regression_logs"`

## Usage

The single digit reader for an image file 1.png `python single_digit_reader.py 1.png`

The multi digit reader for an image file 123.png `python mulit_digit_reader.py 123.png`

