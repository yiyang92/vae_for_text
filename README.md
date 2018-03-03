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
 (or specify through command line parameters). You can choose to train on PTB dataset
  or on Game of Thrones dataset. Parameter debug can be set to true for calling Tensorflow
   debugger.
- --beam_search use beam search

During training generated text will not appear from the first iteration,
be patiant and have fun :)
I will add pre-trained model later.
### Specific requirements
- zhusuan - probabilistic framework https://github.com/thu-ml/zhusuan/
- tensorflow >= 1.0
- gensim (for pretrained w2vec)

### Other files
- train_rnnlm.py - RNN Word-level language model. Can be trained on PTB or
 Game of Thrones dataset.
