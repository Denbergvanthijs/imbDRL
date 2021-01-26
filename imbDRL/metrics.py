import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, roc_curve)
from tensorflow import constant
from tf_agents.trajectories import time_step


def network_predictions(network, X: np.ndarray) -> dict:
    """Computes y_pred using a given network.
    Input is array of data entries.

    :param network: The network to use to calculate metrics
    :type  network: (Q)Network
    :param X: X data, input to network
    :type  X: np.ndarray

    :return: Numpy array of predicted targets for given X
    :rtype: np.ndarray
    """
    if not isinstance(X, np.ndarray):
        raise ValueError(f"`X` must be of type `np.ndarray` not {type(X)}")

    q, _ = network(X, step_type=constant([time_step.StepType.FIRST] * X.shape[0]), training=False)
    return np.argmax(q.numpy(), axis=1)  # Max action for each x in X


def decision_function(network, X: np.ndarray) -> dict:
    """Computes the score for the predicted class of each x in X using a given network.
    Input is array of data entries.

    :param network: The network to use to calculate the score per x in X
    :type  network: (Q)Network
    :param X: X data, input to network
    :type  X: np.ndarray

    :return: Numpy array of scores for given X
    :rtype: np.ndarray
    """
    if not isinstance(X, np.ndarray):
        raise ValueError(f"`X` must be of type `np.ndarray` not {type(X)}")

    q, _ = network(X, step_type=constant([time_step.StepType.FIRST] * X.shape[0]), training=False)
    return np.max(q.numpy(), axis=1)  # Value of max action for each x in X


def classification_metrics(y_true: list, y_pred: list) -> dict:
    """Computes metrics using y_true and y_pred.

    :param y_true: True labels
    :type  y_true: np.ndarray
    :param y_pred: Predicted labels, corresponding to y_true
    :type  y_pred: np.ndarray

    :return: Dictionairy containing Geometric Mean, F1, Precision, Recall, TP, TN, FP, FN
    :rtype: dict
    """
    if not isinstance(y_true, (list, tuple, np.ndarray)):
        raise ValueError(f"`y_true` must be of type `list` not {type(y_true)}")
    if not isinstance(y_pred, (list, tuple, np.ndarray)):
        raise ValueError(f"`y_pred` must be of type `list` not {type(y_pred)}")
    if len(y_true) != len(y_pred):
        raise ValueError("`X` and `y` must be of same length.")

    # labels=[0, 1] to ensure 4 elements are returned: https://stackoverflow.com/a/46230267
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = TP / (TP + FP) if TP + FP else 0  # Positive predictive value
    recall = TP / (TP + FN) if TP + FN else 0  # Sensitivity, True Positive Rate (TPR)
    specificity = TN / (TN + FP) if TN + FP else 0  # Specificity, selectivity, True Negative Rate (TNR)

    G_mean = np.sqrt(recall * specificity)  # Geometric mean of recall and specificity
    F1 = f1_score(y_true, y_pred, zero_division=0)  # Default F-measure

    return {"Gmean": G_mean, "F1": F1, "Precision": precision, "Recall": recall, "TP": TP, "TN": TN, "FP": FP, "FN": FN}


def plot_pr_curve(network, X_test: np.ndarray, y_test: np.ndarray,
                  X_train: np.ndarray = None, y_train: np.ndarray = None) -> None:   # pragma: no cover
    """Plots PR curve of X_test and y_test of given network.
    Optionally plots PR curve of X_train and y_train.
    Average precision is shown in the legend.

    :param network: The network to use to calculate the PR curve
    :type  network: (Q)Network
    :param X_test: X data, input to network
    :type  X_test: np.ndarray
    :param y_test: True labels for `X_test`
    :type  y_test: np.ndarray
    :param X_train: Optional X data to plot validation PR curve
    :type  X_train: np.ndarray
    :param y_train: True labels for `X_val`
    :type  y_train: np.ndarray

    :return: None
    :rtype: NoneType
    """
    plt.plot((0, 1), (1, 0), color="black", linestyle="--", label="Baseline")
    # TODO: Consider changing baseline

    if X_train is not None and y_train is not None:
        y_val_score = decision_function(network, X_train)
        val_precision, val_recall, _ = precision_recall_curve(y_train, y_val_score)
        val_AP = average_precision_score(y_train, y_val_score)
        plt.plot(val_recall, val_precision, label=f"Train AP: {val_AP:.3f}")

    y_test_score = decision_function(network, X_test)
    test_precision, test_recall, _ = precision_recall_curve(y_test, y_test_score)
    test_AP = average_precision_score(y_test, y_test_score)

    plt.plot(test_recall, test_precision, label=f"Test AP: {test_AP:.3f}")
    plt.xlim((-0.05, 1.05))
    plt.ylim((-0.05, 1.05))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()


def plot_roc_curve(network, X_test: np.ndarray, y_test: np.ndarray,
                   X_train: np.ndarray = None, y_train: np.ndarray = None) -> None:   # pragma: no cover
    """Plots ROC curve of X_test and y_test of given network.
    Optionally plots ROC curve of X_train and y_train.
    Average precision is shown in the legend.

    :param network: The network to use to calculate the PR curve
    :type  network: (Q)Network
    :param X_test: X data, input to network
    :type  X_test: np.ndarray
    :param y_test: True labels for `X_test`
    :type  y_test: np.ndarray
    :param X_train: Optional X data to plot validation PR curve
    :type  X_train: np.ndarray
    :param y_train: True labels for `X_val`
    :type  y_train: np.ndarray

    :return: None
    :rtype: NoneType
    """
    plt.plot((0, 1), (0, 1), color="black", linestyle="--", label="Baseline")
    # TODO: Consider changing baseline

    if X_train is not None and y_train is not None:
        y_train_score = decision_function(network, X_train)
        fpr_train, tpr_train, _ = roc_curve(y_train, y_train_score)
        plt.plot(fpr_train, tpr_train, label=f"Train AUROC: {auc(fpr_train, tpr_train):.2f}")

    y_test_score = decision_function(network, X_test)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_score)

    plt.plot(fpr_test, tpr_test, label=f"Test AUROC: {auc(fpr_test, tpr_test):.2f}")
    plt.xlim((-0.05, 1.05))
    plt.ylim((-0.05, 1.05))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


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
    if not all(isinstance(i, (int, np.integer)) for i in (TP, FN, FP, TN)):
        raise ValueError("Not all arguments are integers.")

    ticklabels = ("Minority", "Majority")
    sns.heatmap(((TP, FN), (FP, TN)), annot=True, fmt="_d", cmap="viridis", xticklabels=ticklabels, yticklabels=ticklabels)

    plt.title("Confusion matrix")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.show()
