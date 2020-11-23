# imbDRL

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/Denbergvanthijs/imbDRL/Build) ![License](https://img.shields.io/github/license/Denbergvanthijs/imbDRL)

***Imbalanced Classification with Deep Reinforcement Learning.***

This repository contains multiple implementations of binary classification on unbalanced datasets using [TensorFlow 2.3](https://www.tensorflow.org/) and [TF Agents 0.6](https://www.tensorflow.org/agents):

1. The Double Deep Q-network as published in [this paper](https://arxiv.org/abs/1509.06461) by *van Hasselt et al.* is using a custom environment based on [this paper](https://arxiv.org/abs/1901.01379) by *Lin et al*.

2. The Neural Epsilon Greedy agent is based of [this code](https://www.tensorflow.org/agents/tutorials/bandits_tutorial) from the [TF Agents](https://www.tensorflow.org/agents) team.

Example scripts on the [MNIST](http://yann.lecun.com/exdb/mnist/), [IMDB](http://ai.stanford.edu/~amaas/data/sentiment/) and [Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud) datasets for both implementations can be found in the `./imbDRL/examples` folder.

## Requirements

* [Python 3.8](https://www.python.org/downloads/release/python-386/)
* `pip install -r requirements.txt`
* Optional: `./data/` folder located at the root of this repository.
  * This folder must contain ```creditcard.csv``` downloaded from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) if you would like to use the [Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud) dataset.
  * Note: `creditcard.csv` needs to be split in a seperate train and test file. Please use the function `imbDRL.utils.split_csv`
* Logs will be saved to `./logs/`, trained models will be saved to `./models/`

## Getting started

* For the DDQN examples:
  * `python .\imbDRL\examples\ddqn\train_cartpole.py`
  * `python .\imbDRL\examples\ddqn\train_credit.py`
  * `python .\imbDRL\examples\ddqn\train_image.py`
* For the Bandit examples:
  * `python .\imbDRL\examples\bandit\train_bandit_credit.py`
  * `python .\imbDRL\examples\bandit\train_bandit_image.py`
  * `python .\imbDRL\examples\bandit\train_bandit_imdb.py`

## TensorBoard

To enable [TensorBoard](https://www.tensorflow.org/tensorboard), run ```tensorboard --logdir logs```.

## Tests and linting

Extra arguments are handled with the `./tox.ini` file.

* Pytest: `python -m pytest`
* Flake8: `flake8`
* Coverage can be found in the `./htmlcov` folder
