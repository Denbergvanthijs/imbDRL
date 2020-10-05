import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10, fashion_mnist, imdb, mnist
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_image(data_source: str):
    """
    Loads one of the following image datasets: {mnist, famnist, cifar10}.
    Normalizes the data. Returns X and y.
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

    X = np.concatenate((X_train, X_test))  # Combine train/test to make new train/test/validate later on
    y = np.concatenate((y_train, y_test))

    X = X.reshape(reshape_shape)
    X = X / 255  # /= is not available when casting int to float: https://stackoverflow.com/a/48948461/10603874
    y = y.reshape(y.shape[0], )

    return X, y


def load_imdb(config=(5_000, 500)):
    """Loads the IMDB dataset. Returns X, y."""
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=config[0])
    X = np.concatenate((X_train, X_test))  # Combine train/test to make new train/test/validate later on
    y = np.concatenate((y_train, y_test))

    X = pad_sequences(X, maxlen=config[1])

    return X, y


def load_creditcard(fp: str = "./data/creditcard.csv"):
    """
    Loads the Kaggle Credit Card Fraud dataset from local filepath. Returns X, y.
    Source: https://www.kaggle.com/mlg-ulb/creditcardfraud
    """
    X = read_csv(fp)  # Directly converted to float64

    y = X["Class"]  # 1: Fraud/Minority, 0: No fraud/Majority
    X.drop(columns=["Time", "Class"], inplace=True)  # Dropping `Time` since future data for the model could have another epoch

    return X.values, y.values  # Numpy arrays


def load_data(data_source: str, imb_rate: float, min_class: list, maj_class: list,
              seed: float = None, normalization: bool = False, print_stats: bool = True):
    """
    Loads data from the `data_source`. Imbalances the data and divides the data into train, test and validation sets.
    The imbalance rate of each individual dataset is the same as the given `imb_rate`.
    The seed for the test dataset is already set to ensure test data is always the same.
    """
    if data_source in ("famnist", "mnist", "cifar10"):
        X, y = load_image(data_source=data_source)
    elif data_source == "credit":
        X, y = load_creditcard()
    elif data_source == "imdb":
        X, y = load_imdb()
    else:
        raise ValueError("No valid `data_source`.")

    X_imb, y_imb = get_imb_data(X, y, imb_rate, min_class, maj_class)  # Imbalance the data

    # 60 / 20 / 20 for train / test / validate; stratify=y to ensure class balance is kept
    # Seed for train / test is always 42 to ensure test data is always the same
    # Seed for train / validate is not set to ensure random split
    X_rest, X_test, y_rest, y_test = train_test_split(X_imb, y_imb, test_size=0.2, random_state=42, stratify=y_imb)
    X_train, X_val, y_train, y_val = train_test_split(X_rest, y_rest, test_size=0.25, random_state=seed, stratify=y_rest)

    if data_source == "credit" and normalization:
        # Normalize data. This does not happen in load_creditcard to prevent train/test/val leakage
        # Other data sources are already normalized. RGB values are always in range 0 to 255.
        mini, maxi = X_train.min(axis=0), X_train.max(axis=0)
        for X in (X_train, X_test, X_val):  # Normalize to min-max
            X -= mini
            X /= maxi-mini

    if print_stats:
        p_data, p_train, p_test, p_val = [((y == 1).sum(), (y == 1).sum() / (y == 0).sum()) for y in (y_imb, y_train, y_test, y_val)]
        print(f"Imbalance ratio `p`:\n"
              f"\tdataset:    n={p_data[0]}, p={p_data[1]:.6f}\n"
              f"\ttrain:      n={p_train[0]}, p={p_train[1]:.6f}\n"
              f"\ttest:       n={p_test[0]}, p={p_test[1]:.6f}\n"
              f"\tvalidation: n={p_val[0]}, p={p_val[1]:.6f}")

    return X_train, y_train, X_test, y_test, X_val, y_val


def get_imb_data(X, y, imb_rate: float, min_class: list, maj_class: list):
    """
    Split data in minority and majority, only values in {min_class, maj_class} will be kept.
    (Possibly) decrease minority rows to match the imbalance rate.
    If initial imb_rate of dataset is lower than given `imb_rate`, the imb_rate will not be changed.
    Labels of minority and majority will change to 1 and 0.

    Note: Data will not be shuffled
    """
    X_min, X_maj = [], []

    for i, value in enumerate(y):
        if value in min_class:
            X_min.append(X[i])

        if value in maj_class:
            X_maj.append(X[i])

    X_maj_len = len(X_maj)
    min_len = int(X_maj_len * imb_rate)

    # Keep all majority rows, decrease minority rows to match `imb_rate`
    X_imb = np.array(X_maj + X_min[:min_len])  # `min_len` can be more than the number of minority rows
    y_imb = np.concatenate((np.zeros(X_maj_len), np.ones(X_imb.shape[0] - X_maj_len)))

    return X_imb, y_imb


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, X_val, y_val = load_data("credit", 0.01, [1], [0])
    print([i.shape for i in (X_train, y_train, X_test, y_test, X_val, y_val)])
