import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split


def plot_confusion_matrix(TP: int, FN: int, FP: int, TN: int) -> None:  # pragma: no cover
    """Plots confusion matric of given TP, FN, FP, TN.

    :param TP: True Positive
    :type  TP: int
    :param FN: False Negative
    :type  FN: int
    :param FP: False Positive
    :type  FP: int
    :param TN: True Negative
    :type  TN: int

    :return: None
    :rtype: NoneType
    """
    if not all(isinstance(i, int) for i in (TP, FN, FP, TN)):
        raise ValueError("Not all arguments are integers.")

    ticklabels = ("Minority", "Majority")
    sns.heatmap(((TP, FN), (FP, TN)), annot=True, fmt="_d", cmap="viridis", xticklabels=ticklabels, yticklabels=ticklabels)

    plt.title("Confusion matrix")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.show()


def split_csv(fp: str = "./data/creditcard.csv", fp_dest: str = "./data",
              name: str = "credit", test_size: int = 0.5, strat_col: str = "Class") -> None:
    """Splits a csv file in two, in a stratified fashion.
    Format for filenames will be `{name}0.csv and `{name}1.csv`.

    :param fp: The path at which the csv file is located.
    :type  fp: str
    :param fp_dest: The path to save the train and test files.
    :type  fp_dest: str
    :param name: The prefix for the files.
    :type  name: str
    :param test_size: The fraction of total size for the test file.
    :type  test_size: float
    :param strat_col: The column in the original csv file to stratify.

    :return: None, two files located at `fp_dest`.
    :rtype: NoneType
    """
    if not os.path.isfile(fp):
        raise FileNotFoundError(f"File at {fp=} does not exist.")
    if not os.path.isdir(fp_dest):
        raise ValueError(f"Directory at {fp_dest=} does not exist.")
    if not 0 < test_size < 1:
        raise ValueError(f"{test_size} is not in interval 0 < x < 1.")

    df = pd.read_csv(fp)

    if not (strat_col in df.columns):
        raise ValueError(f"Stratify column {strat_col} not found in DataFrame.")

    train, test = train_test_split(df, test_size=test_size, stratify=df[strat_col])

    train.to_csv(f"{fp_dest}/{name}0.csv", index=False)
    test.to_csv(f"{fp_dest}/{name}1.csv", index=False)


def get_reward_distribution(imb_rate: float):
    """Generates reward distribution based on given imbalance rate.

    From the documentation of tf_agents.bandits.environments.ClassificationBanditEnvironment:

    reward_distribution: a `tfd.Distribution` with event_shape
        `[num_classes, num_actions]`. Entry `[i, j]` is the reward for taking
        action `j` for an instance of class `i`.

    The environment will always have 2 classes with each 2 actions.
    Array[0][0] corresponds to action 0 on class 0, thus the majority class, thus True Negative
    Array[0][1]: False Positive reward
    Array[1][0]: False Negative reward
    Array[1][1]: True Positive reward

    :param imb_rate: The imbalance rate of the data: len(min_class) / len(maj_class)
    :type  imb_rate: float

    :return: Combined Independent distribution
    :rtype: tensorflow_probability.Distribution
    """
    if not 0 < imb_rate < 1:
        raise ValueError(f"{imb_rate} is not in interval 0 < x < 1.")

    bernoulli = tfp.distributions.Bernoulli(probs=[[0, 0], [0, 0]], dtype=np.float32)
    combined = (tfp.bijectors.Shift([[imb_rate, -imb_rate], [-1, 1]])
                (tfp.bijectors.Scale([[1, 1], [1, 1]])
                 (bernoulli)))
    return tfp.distributions.Independent(combined, reinterpreted_batch_ndims=2)


def rounded_dict(d: dict, precision: int = 6) -> dict:
    """Rounds all values in a dictionairy to `precision` digits after the decimal point.

    :param d: Dictionairy containing only floats or ints as values
    :type  d: dict

    :return: Rounded dictionairy
    :rtype: dict
    """
    return {k: round(v, precision) for k, v in d.items()}
