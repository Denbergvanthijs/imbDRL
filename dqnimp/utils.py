import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


def plot_confusion_matrix(TP: int, FN: int, FP: int, TN: int) -> None:
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
