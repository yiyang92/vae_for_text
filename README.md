# Using Variational Auto-Encoder For Generating Texts
## Overview

Tensorflow implementation of [Generating Sentences from a Continuous Space](https://arxiv.org/abs/1511.06349).

## Usage

Training:
```shell=
python vae_lstm-lstm.py
```
 ### Parameters
 Parameters can be set directly in Parameters class in vae_lstm-lst.py file.
 (later will change it to command line entries). You can choose to train on PTB dataset
  or on Game of Thrones dataset. Parameter debug can be set to true for calling Tensorflow
   debugger.

### Other files
- train_rnnlm.py - RNN Word-level language model. Can be trained on PTB or
 Game of Thrones dataset.
- dilated_conv_mnist_test.py - some playground for learning dilated CNN 
parameters tuning. Will be deleted after vae-lstm-cnn code initial release
 
 
 
 
 
 
 
 
 
 
 
 
 
