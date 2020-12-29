import os

from imbDRL.agents.ddqn import TrainDDQN
from imbDRL.data import load_creditcard
from imbDRL.metrics import (classification_metrics, network_predictions,
                            plot_pr_curve, plot_roc_curve)
from imbDRL.utils import rounded_dict

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU is faster than GPU

min_class = [1]  # Minority classes, same setup as in original paper
maj_class = [0]  # Majority classes
fp_model = "./models/20201229_131626.pkl"

X_train, y_train, X_test, y_test, = load_creditcard(normalization=True)
network = TrainDDQN.load_model(fp_model)

y_pred_train = network_predictions(network, X_train)
y_pred_test = network_predictions(network, X_test)

stats = classification_metrics(y_train, y_pred_train)
print(f"Train: {rounded_dict(stats)}")
stats = classification_metrics(y_test, y_pred_test)
print(f"Test:  {rounded_dict(stats)}")

plot_pr_curve(network, X_test, y_test, X_train, y_train)
plot_roc_curve(network, X_test, y_test, X_train, y_train)
