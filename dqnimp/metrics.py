import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score
from tf_agents.trajectories import time_step as ts


def metrics_by_network(network, X: list, y: list) -> dict:
    """Computes metrics using a given network."""
    q, _ = network(X)
    y_pred = np.argmax(q.numpy(), axis=1)  # Max action for each x in X

    return classify_metrics(y, y_pred)


def metrics_by_policy(X_val: list, y_val: list, policy) -> dict:
    """Computes metrics using a given policy."""
    y_pred = []

    for x in X_val:
        tf_x = tf.constant(x, shape=(1, 29))
        time_step = ts.transition(observation=tf_x, reward=0.0, outer_dims=(1, 29,))
        action_step = policy.action(time_step)
        y_pred.append(action_step.action)

    y_pred = tf.stack(y_pred)
    return classify_metrics(y_val, y_pred)


def classify_metrics(y_true: list, y_pred: list) -> dict:
    """Computes metrics using y_true and y_pred."""
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()

    recall = TP / denom if (denom := TP + FN) else 0  # Sensitivity, True Positive Rate (TPR)
    specificity = TN / denom if (denom := TN + FP) else 0  # Specificity, selectivity, True Negative Rate (TNR)

    G_mean = np.sqrt(recall * specificity)  # Geometric mean of recall and specificity
    Fdot5 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)  # β of 0.5
    F1 = f1_score(y_true, y_pred, zero_division=0)  # Default F-measure
    F2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)  # β of 2

    return {"Gmean": G_mean, "Fdot5": Fdot5, "F1": F1, "F2": F2, "TP": TP, "TN": TN, "FP": FP, "FN": FN}


if __name__ == "__main__":
    import pickle

    from dqnimp.data import load_data

    imb_rate = 0.00173  # Imbalance rate
    min_class = [1]  # Minority classes, must be same as trained model
    maj_class = [0]  # Majority classes, must be same as trained model
    datasource = "credit"  # The dataset to be selected
    _, _, X_test, y_test, _, _ = load_data(datasource, imb_rate, min_class, maj_class)  # Load all data

    with open("./models/20201007_121850.pkl", "rb") as f:  # Load the Q-network
        network = pickle.load(f)

    metrics = metrics_by_network(network, X_test, y_test)
