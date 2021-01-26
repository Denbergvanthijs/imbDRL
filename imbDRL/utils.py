import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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
        raise FileNotFoundError(f"File at {fp} does not exist.")
    if not os.path.isdir(fp_dest):
        raise ValueError(f"Directory at {fp_dest} does not exist.")
    if not 0 < test_size < 1:
        raise ValueError(f"{test_size} is not in interval 0 < x < 1.")

    df = pd.read_csv(fp)

    if not (strat_col in df.columns):
        raise ValueError(f"Stratify column {strat_col} not found in DataFrame.")

    train, test = train_test_split(df, test_size=test_size, stratify=df[strat_col])

    train.to_csv(f"{fp_dest}/{name}0.csv", index=False)
    test.to_csv(f"{fp_dest}/{name}1.csv", index=False)


def rounded_dict(d: dict, precision: int = 6) -> dict:
    """Rounds all values in a dictionairy to `precision` digits after the decimal point.

    :param d: Dictionairy containing only floats or ints as values
    :type  d: dict

    :return: Rounded dictionairy
    :rtype: dict
    """
    return {k: round(v, precision) for k, v in d.items()}


def imbalance_ratio(y: np.ndarray, min_classes: List[int] = [1], maj_classes: List[int] = [0]) -> float:
    """Calculates imbalance ratio of minority class(es) and majority class(es).

    :param y: y-vector with labels.
    :type  y: np.ndarray
    :param min_classes: The labels of the minority classes
    :type  min_classes: list
    :param maj_classes: The labels of the minority classes
    :type  maj_classes: list

    :return: The imbalance ratio
    :rtype: float
    """
    return np.isin(y, min_classes).sum() / np.isin(y, maj_classes).sum()
