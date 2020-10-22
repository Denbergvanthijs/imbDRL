import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score
from tf_agents.trajectories import time_step as ts


def metrics_by_network(network, X: np.ndarray, y: np.ndarray) -> dict:
    """Computes metrics using a given network.
    Input is X and y data. X will be used to make y_pred.
    y_pred and y will be compared using `imbDRL.metrics.classification_metrics()`.

    :param network: The network to use to calculate metrics
    :type  network: (Q)Network
    :param X: X data
    :type  X: np.ndarray
    :param y: y data, labels corresponding to X
    :type  y: np.ndarray

    :return: Dictionairy containing Geometric Mean, F0.5, F1, F2, TP, TN, FP, FN
    :rtype: dict
    """
    if not isinstance(X, np.ndarray):
        raise ValueError(f"`X` must be of type `np.ndarray` not {type(X)}")
    if not isinstance(y, np.ndarray):
        raise ValueError(f"`y` must be of type `np.ndarray` not {type(y)}")
    if X.shape[0] != y.shape[0]:
        raise ValueError("`X` and `y` must contain the same amount of rows.")

    q, _ = network(X)
    y_pred = np.argmax(q.numpy(), axis=1)  # Max action for each x in X

    return classification_metrics(y, y_pred)


def metrics_by_policy(policy, X: list, y: list) -> dict:
    """Computes metrics using a given policy.
    Input is X and y data. X will be used to make y_pred.
    y_pred and y will be compared using `imbDRL.metrics.classification_metrics()`.

    :param policy: The policy to use to calculate metrics
    :type  policy: policy
    :param X: X data
    :type  X: np.ndarray
    :param y: y data, labels corresponding to X
    :type  y: np.ndarray

    :return: Dictionairy containing Geometric Mean, F0.5, F1, F2, TP, TN, FP, FN
    :rtype: dict
    """
    if not isinstance(X, np.ndarray):
        raise ValueError(f"`X` must be of type `np.ndarray` not {type(X)}")
    if not isinstance(y, np.ndarray):
        raise ValueError(f"`y` must be of type `np.ndarray` not {type(y)}")
    if X.shape[0] != y.shape[0]:
        raise ValueError("`X` and `y` must contain the same amount of rows.")

    y_pred = []
    shape = (1, ) + X.shape[1:]
    for x in X:
        tf_x = tf.constant(x, shape=shape)
        time_step = ts.transition(observation=tf_x, reward=0.0, outer_dims=shape)
        action_step = policy.action(time_step)
        y_pred.append(action_step.action)

    return classification_metrics(y, tf.stack(y_pred))


def classification_metrics(y_true: list, y_pred: list) -> dict:
    """Computes metrics using y_true and y_pred.

    :param y_true: True labels
    :type  y_true: np.ndarray
    :param y_pred: Predicted labels, corresponding to y_true
    :type  y_pred: np.ndarray

    :return: Dictionairy containing Geometric Mean, F0.5, F1, F2, TP, TN, FP, FN
    :rtype: dict
    """
    if not isinstance(y_true, (list, tuple, np.ndarray)):
        raise ValueError(f"`y_true` must be of type `list` not {type(y_true)}")
    if not isinstance(y_pred, (list, tuple, np.ndarray)):
        raise ValueError(f"`y_pred` must be of type `list` not {type(y_pred)}")
    if len(y_true) != len(y_pred):
        raise ValueError("`X` and `y` must be of same length.")

    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()

    recall = TP / denom if (denom := TP + FN) else 0  # Sensitivity, True Positive Rate (TPR)
    specificity = TN / denom if (denom := TN + FP) else 0  # Specificity, selectivity, True Negative Rate (TNR)

    G_mean = np.sqrt(recall * specificity)  # Geometric mean of recall and specificity
    Fdot5 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)  # β of 0.5
    F1 = f1_score(y_true, y_pred, zero_division=0)  # Default F-measure
    F2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)  # β of 2

    return {"Gmean": G_mean, "Fdot5": Fdot5, "F1": F1, "F2": F2, "TP": TP, "TN": TN, "FP": FP, "FN": FN}
