# DQNimp

Deep Q Network, improved.

This repository contains an implementation of binary classification on unbalanced datasets using [TensorFlow 2.3](https://www.tensorflow.org/) and [TF Agents](https://www.tensorflow.org/agents).

The classification model uses a DDQN as published in [this paper](https://arxiv.org/abs/1509.06461) from van Hasselt et al. with an custom environment based on the [paper](https://arxiv.org/abs/1901.01379) of Lin et al.

## Requirements

* [Python 3.8](https://www.python.org/downloads/release/python-386/)
* To make a new environment from the included `env.yml`-file, please run ```conda env create -f env.yml```
  * To save your conda environment use ```conda env export > env.yml```
* A ```./data``` folder located at the root of this repository. This folder must contain the ```creditcard.csv``` file from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) if you would like to use the Credit Card Fraud Dataset.

## Getting started

To start training, run ```python train.py``` in the terminal after activating your conda environment.

To enable [TensorBoard](https://www.tensorflow.org/tensorboard), run ```tensorboard --logdir logs```.

## Contributions

Please run Pytest and Flake8 before any Pull Request:

* ```pytest -vs```
* ```flake8 . --ignore=D100,D104,D205,D401,I100,I201 --show-source --enable-extension=G --max-line-length=140 --max-complexity=10 --count```
