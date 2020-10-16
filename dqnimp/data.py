import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10, fashion_mnist, imdb, mnist
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tf_agents.trajectories import trajectory
from tqdm import tqdm


def load_image(data_source: str):
    """
    Loads one of the following image datasets: {mnist, famnist, cifar10}.
    Normalizes the data. Returns X and y for both train and test datasets.
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


def load_imdb(config=(5_000, 500)):
    """Loads the IMDB dataset. Returns X and y for both train and test datasets."""
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=config[0])

    X_train = pad_sequences(X_train, maxlen=config[1])
    X_test = pad_sequences(X_test, maxlen=config[1])

    return X_train, y_train, X_test, y_test


def load_creditcard(fp_train: str = "./data/credit0.csv", fp_test: str = "./data/credit1.csv", normalization: bool = False):
    """
    Loads the Kaggle Credit Card Fraud dataset from local filepath. Returns X and y for both train and test datasets.
    Source: https://www.kaggle.com/mlg-ulb/creditcardfraud
    """
    X_train = read_csv(fp_train).astype(np.float32)  # DataFrames directly converted to float32
    X_test = read_csv(fp_test).astype(np.float32)

    y_train = X_train["Class"].astype(np.int32)  # 1: Fraud/Minority, 0: No fraud/Majority
    y_test = X_test["Class"].astype(np.int32)
    X_train.drop(columns=["Time", "Class"], inplace=True)  # Dropping `Time` since future data for the model could have another epoch
    X_test.drop(columns=["Time", "Class"], inplace=True)

    # Other data sources are already normalized. RGB values are always in range 0 to 255.
    if normalization:
        mini, maxi = X_train.min(axis=0), X_train.max(axis=0)
        X_train -= mini
        X_train /= maxi-mini
        X_test -= mini
        X_test /= maxi-mini

    return X_train.values, y_train.values, X_test.values, y_test.values  # Numpy arrays


def load_data(data_source: str, imb_rate: float, min_class: list, maj_class: list, normalization: bool = False, print_stats: bool = True):
    """
    Loads data from the `data_source`. Imbalances the data and divides the data into train, test and validation sets.
    The imbalance rate of each individual dataset is the same as the given `imb_rate`.
    The seed for the test dataset is already set to ensure test data is always the same.
    """
    if data_source in ("famnist", "mnist", "cifar10"):
        X_train, y_train, X_test, y_test = load_image(data_source=data_source)
    elif data_source == "credit":
        X_train, y_train, X_test, y_test = load_creditcard(normalization=normalization)
    elif data_source == "imdb":
        X_train, y_train, X_test, y_test = load_imdb()
    else:
        raise ValueError("No valid `data_source`.")

    X_train, y_train = get_imb_data(X_train, y_train, imb_rate, min_class, maj_class)  # Imbalance the data
    X_test, y_test = get_imb_data(X_test, y_test, imb_rate, min_class, maj_class)  # Imbalance the data

    # 60 / 20 / 20 for train / test / validate; stratify=y to ensure class balance is kept
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train)

    if print_stats:
        p_train, p_test, p_val = [((y == 1).sum(), (y == 1).sum() / (y == 0).sum()) for y in (y_train, y_test, y_val)]
        print(f"Imbalance ratio `p`:\n"
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

    min_len = int((X_maj_len := len(X_maj)) * imb_rate)

    # Keep all majority rows, decrease minority rows to match `imb_rate`
    X_imb = np.array(X_maj + X_min[:min_len], dtype=np.float32)  # `min_len` can be more than the number of minority rows
    y_imb = np.concatenate((np.zeros(X_maj_len), np.ones(X_imb.shape[0] - X_maj_len))).astype(np.int32)

    return X_imb, y_imb


def collect_step(environment, policy, buffer) -> None:
    """Data collection for 1 step."""
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps: int, logging: bool = False) -> None:
    """Collect data for a number of steps. Mainly used for warmup period."""
    if logging:
        for _ in tqdm(range(steps)):
            collect_step(env, policy, buffer)
    else:
        for _ in range(steps):
            collect_step(env, policy, buffer)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, X_val, y_val = load_data("credit", 0.00173, [1], [0])
    print(*[i.dtype for i in (X_train, y_train, X_test, y_test, X_val, y_val)])
    print(*[i.shape for i in (X_train, y_train, X_test, y_test, X_val, y_val)])
