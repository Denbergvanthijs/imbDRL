import os
from typing import List, Tuple

import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.datasets import cifar10, fashion_mnist, imdb, mnist
from tensorflow.keras.preprocessing.sequence import pad_sequences

from imbDRL.utils import imbalance_ratio

TrainTestData = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
TrainTestValData = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def load_image(data_source: str) -> TrainTestData:
    """
    Loads one of the following image datasets: {mnist, famnist, cifar10}.
    Normalizes the data. Returns X and y for both train and test datasets.
    Dtypes of X's and y's will be `float32` and `int32` to be compatible with `tf_agents`.

    :param data_source: Either mnist, famnist or cifar10
    :type  data_source: str

    :return: Tuple of (X_train, y_train, X_test, y_test) containing original split of train/test
    :rtype: tuple
    """
    reshape_shape = -1, 28, 28, 1

    if data_source == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    elif data_source == "famnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    elif data_source == "cifar10":
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        reshape_shape = -1, 32, 32, 3

    else:
        raise ValueError("No valid `data_source`.")

    X_train = X_train.reshape(reshape_shape).astype(np.float32)  # Float32 is the expected dtype for the observation spec in the env
    X_test = X_test.reshape(reshape_shape).astype(np.float32)

    X_train /= 255  # /= is not available when casting int to float: https://stackoverflow.com/a/48948461/10603874
    X_test /= 255

    y_train = y_train.reshape(y_train.shape[0], ).astype(np.int32)
    y_test = y_test.reshape(y_test.shape[0], ).astype(np.int32)

    return X_train, y_train, X_test, y_test


def load_csv(fp_train: str, fp_test: str, label_col: str, drop_cols: List[str], normalization: bool = False) -> TrainTestData:
    """
    Loads any csv-file from local filepaths. Returns X and y for both train and test datasets.
    Option to normalize the data with min-max normalization.
    Only csv-files with float32 values for the features and int32 values for the labels supported.
    Source for dataset: https://mimic-iv.mit.edu/

    :param fp_train: Location of the train csv-file
    :type  fp_train: str
    :param fp_test: Location of the test csv-file
    :type  fp_test: str
    :param label_col: The name of the column containing the labels of the data
    :rtype label_col: str
    :param drop_cols: List of the names of the columns to be dropped. `label_col` gets dropped automatically
    :rtype drop_cols: List of strings
    :param normalization: Normalize the data with min-max normalization?
    :type  normalization: bool

    :return: Tuple of (X_train, y_train, X_test, y_test) containing original split of train/test
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    if not os.path.isfile(fp_train):
        raise FileNotFoundError(f"`fp_train` {fp_train} does not exist.")
    if not os.path.isfile(fp_test):
        raise FileNotFoundError(f"`fp_test` {fp_test} does not exist.")
    if not isinstance(normalization, bool):
        raise TypeError(f"`normalization` must be of type `bool`, not {type(normalization)}")

    X_train = read_csv(fp_train).astype(np.float32)  # DataFrames directly converted to float32
    X_test = read_csv(fp_test).astype(np.float32)

    y_train = X_train[label_col].astype(np.int32)
    y_test = X_test[label_col].astype(np.int32)
    X_train.drop(columns=drop_cols + [label_col], inplace=True)  # Dropping cols and label column
    X_test.drop(columns=drop_cols + [label_col], inplace=True)

    # Other data sources are already normalized. RGB values are always in range 0 to 255.
    if normalization:
        mini, maxi = X_train.min(axis=0), X_train.max(axis=0)
        X_train -= mini
        X_train /= maxi - mini
        X_test -= mini
        X_test /= maxi - mini

    return X_train.values, y_train.values, X_test.values, y_test.values  # Numpy arrays


def load_imdb(config: Tuple[int, int] = (5_000, 500)) -> TrainTestData:
    """Loads the IMDB dataset. Returns X and y for both train and test datasets.

    :param config: Tuple of number of most frequent words and max length of each sequence.
    :type  config: str

    :return: Tuple of (X_train, y_train, X_test, y_test) containing original split of train/test
    :rtype: tuple
    """
    if not isinstance(config, (tuple, list)):
        raise TypeError(f"{type(config)} is no valid datatype for `config`.")
    if len(config) != 2:
        raise ValueError("Tuple length of `config` must be 2.")
    if not all(i > 0 for i in config):
        raise ValueError("All integers of `config` must be > 0.")

    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=config[0])

    X_train = pad_sequences(X_train, maxlen=config[1])
    X_test = pad_sequences(X_test, maxlen=config[1])

    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    return X_train, y_train, X_test, y_test


def get_train_test_val(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, min_classes: List[int],
                       maj_classes: List[int], imb_ratio: float = None, imb_test: bool = True, val_frac: float = 0.25,
                       print_stats: bool = True) -> TrainTestValData:
    """
    Imbalances data and divides the data into train, test and validation sets.
    The imbalance rate of each individual dataset is approx. the same as the given `imb_ratio`.

    :param X_train: The X_train data
    :type  X_train: np.ndarray
    :param y_train: The y_train data
    :type  y_train: np.ndarray
    :param X_test: The X_test data
    :type  X_test: np.ndarray
    :param y_test: The y_test data
    :type  y_test: np.ndarray
    :param min_classes: List of labels of all minority classes
    :type  min_classes: list
    :param maj_classes: List of labels of all majority classes.
    :type  maj_classes: list
    :param imb_ratio: Imbalance ratio for minority to majority class: len(minority datapoints) / len(majority datapoints)
        If the `imb_ratio` is None, data will not be imbalanced and will only be relabeled to 1's and 0's.
    :type  imb_ratio: float
    :param imb_test: Imbalance the test dataset?
    :type  imb_test: bool
    :param val_frac: Fraction to take from X_train and y_train for X_val and y_val
    :type  val_frac: float
    :param print_stats: Print the imbalance ratio of the imbalanced data?
    :type  print_stats: bool

    :return: Tuple of (X_train, y_train, X_test, y_test, X_val, y_val)
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    if not 0 < val_frac < 1:
        raise ValueError(f"{val_frac} is not in interval 0 < x < 1.")
    if not isinstance(print_stats, bool):
        raise TypeError(f"`print_stats` must be of type `bool`, not {type(print_stats)}.")

    X_train, y_train = imbalance_data(X_train, y_train, min_classes, maj_classes, imb_ratio=imb_ratio)
    # Only imbalance test-data if imb_test is True
    X_test, y_test = imbalance_data(X_test, y_test, min_classes, maj_classes, imb_ratio=imb_ratio if imb_test else None)

    # stratify=y_train to ensure class balance is kept between train and validation datasets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_frac, stratify=y_train)

    if print_stats:
        p_train, p_test, p_val = [((y == 1).sum(), imbalance_ratio(y)) for y in (y_train, y_test, y_val)]
        print(f"Imbalance ratio `p`:\n"
              f"\ttrain:      n={p_train[0]}, p={p_train[1]:.6f}\n"
              f"\ttest:       n={p_test[0]}, p={p_test[1]:.6f}\n"
              f"\tvalidation: n={p_val[0]}, p={p_val[1]:.6f}")

    return X_train, y_train, X_test, y_test, X_val, y_val


def imbalance_data(X: np.ndarray, y: np.ndarray, min_class: List[int], maj_class: List[int],
                   imb_ratio: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split data in minority and majority, only values in {min_class, maj_class} will be kept.
    (Possibly) decrease minority rows to match the imbalance rate.
    If initial imb_ratio of dataset is lower than given `imb_ratio`, the imb_ratio of the returned data will not be changed.
    If the `imb_ratio` is None, data will not be imbalanced and will only be relabeled to 1's and 0's.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError(f"`X` must be of type `np.ndarray` not {type(X)}")
    if not isinstance(y, np.ndarray):
        raise TypeError(f"`y` must be of type `np.ndarray` not {type(y)}")
    if X.shape[0] != y.shape[0]:
        raise ValueError("`X` and `y` must contain the same amount of rows.")
    if not isinstance(min_class, (list, tuple)):
        raise TypeError("`min_class` must be of type list or tuple.")
    if not isinstance(maj_class, (list, tuple)):
        raise TypeError("`maj_class` must be of type list or tuple.")

    if (imb_ratio is not None) and not (0 < imb_ratio < 1):
        raise ValueError(f"{imb_ratio} is not in interval 0 < imb_ratio < 1.")

    if imb_ratio is None:  # Do not imbalance data if no `imb_ratio` is given
        imb_ratio = 1

    X_min = X[np.isin(y, min_class)]  # Mask the correct indexes
    X_maj = X[np.isin(y, maj_class)]  # Only keep data/labels for x in {min_class, maj_class} and forget all other

    min_len = int(X_maj.shape[0] * imb_ratio)  # Amount of rows to select from minority classes to get to correct imbalance ratio
    # Keep all majority rows, decrease minority rows to match `imb_ratio`
    X_min = X_min[np.random.choice(X_min.shape[0], min(min_len, X_min.shape[0]), replace=False), :]

    X_imb = np.concatenate([X_maj, X_min]).astype(np.float32)
    y_imb = np.concatenate((np.zeros(X_maj.shape[0]), np.ones(X_min.shape[0]))).astype(np.int32)
    X_imb, y_imb = shuffle(X_imb, y_imb)

    return X_imb, y_imb
