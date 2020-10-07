import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score
from tf_agents.trajectories import trajectory


def compute_metrics(network, X, y):
    """Computes the confusion matrix for a given dataset."""
    q, _ = network(X)
    y_pred = np.argmax(q.numpy(), axis=1)  # Max action for each x in X
    TN, FP, FN, TP = confusion_matrix(y, y_pred).ravel()

    recall = TP / denom if (denom := TP + FN) else 0  # Sensitivity, True Positive Rate (TPR)
    specificity = TN / denom if (denom := TN + FP) else 0  # Specificity, selectivity, True Negative Rate (TNR)

    G_mean = np.sqrt(recall * specificity)  # Geometric mean of recall and specificity
    Fdot5 = fbeta_score(y, y_pred, beta=0.5, zero_division=0)  # β of 0.5
    F1 = f1_score(y, y_pred, zero_division=0)  # Default F-measure
    F2 = fbeta_score(y, y_pred, beta=2, zero_division=0)  # β of 2

    return {"Gmean": G_mean, "Fdot5": Fdot5, "F1": F1, "F2": F2, "TP": TP, "TN": TN, "FP": FP, "FN": FN}


def collect_step(environment, policy, buffer):
    """Data collection for 1 step."""
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps: int):
    """Collect data for a number of steps. Mainly used for warmup period."""
    for _ in range(steps):
        collect_step(env, policy, buffer)


if __name__ == "__main__":
    import pickle

    from get_data import load_data

    imb_rate = 0.00173  # Imbalance rate
    min_class = [1]  # Minority classes, must be same as trained model
    maj_class = [0]  # Majority classes, must be same as trained model
    datasource = "credit"  # The dataset to be selected
    _, _, X_test, y_test, _, _ = load_data(datasource, imb_rate, min_class, maj_class)  # Load all data

    with open("./models/20201007_121850.pkl", "rb") as f:  # Load the Q-network
        network = pickle.load(f)

    metrics = compute_metrics(network, X_test, y_test)

    print(*[(k, round(v, 6)) for k, v in metrics.items()])
