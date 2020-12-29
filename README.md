# imbDRL

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/Denbergvanthijs/imbDRL/Build) ![License](https://img.shields.io/github/license/Denbergvanthijs/imbDRL)

***Imbalanced Classification with Deep Reinforcement Learning.***

This repository contains an (Double) Deep Q-Network implementation of binary classification on unbalanced datasets using [TensorFlow 2.3 / 2.4](https://www.tensorflow.org/) and [TF Agents 0.6](https://www.tensorflow.org/agents):

* The Double Deep Q-network as published in [this paper](https://arxiv.org/abs/1509.06461) by *van Hasselt et al.* is using a custom environment based on [this paper](https://arxiv.org/abs/1901.01379) by *Lin et al*.

Example scripts on the [Mnist](http://yann.lecun.com/exdb/mnist/), [Fashion Mnist](https://github.com/zalandoresearch/fashion-mnist) and [Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud) datasets can be found in the `./imbDRL/examples/ddqn/` folder.

## Requirements

* [Python 3.7+](https://www.python.org/)
* `pip install -r requirements.txt`
* Logs are by default saved in `./logs/`
* Trained models are by default saved in `./models/`
* Optional: `./data/` folder located at the root of this repository.
  * This folder must contain ```creditcard.csv``` downloaded from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) if you would like to use the [Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud) dataset.
  * Note: `creditcard.csv` needs to be split in a seperate train and test file. Please use the function `imbDRL.utils.split_csv`

## Getting started

Run any of the following scripts:

* `python .\imbDRL\examples\ddqn\train_credit.py`
* `python .\imbDRL\examples\ddqn\train_famnist.py`
* `python .\imbDRL\examples\ddqn\train_mnist.py`

## TensorBoard

To enable [TensorBoard](https://www.tensorflow.org/tensorboard), run ```tensorboard --logdir logs```

## Tests and linting

Extra arguments are handled with the `./tox.ini` file.

* Pytest: `python -m pytest`
* Flake8: `flake8`
* Coverage can be found in the generated `./htmlcov` folder
