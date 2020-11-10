from imbDRL.data import get_train_test_val, load_creditcard
from imbDRL.examples.ddqn.example_classes import TrainCustomDDQN
from imbDRL.metrics import (classification_metrics, network_predictions,
                            plot_pr_curve, plot_roc_curve)
from imbDRL.utils import rounded_dict

min_class = [1]  # Minority classes, same setup as in original paper
maj_class = [0]  # Majority classes
fp_model = "./models/20201110_125147"

X_train, y_train, X_test, y_test, = load_creditcard(normalization=True)
X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_val(X_train, y_train, X_test, y_test, min_class, maj_class)

network = TrainCustomDDQN.load_model(fp_model)

y_pred_train = network_predictions(network, X_train)
y_pred_val = network_predictions(network, X_val)
y_pred_test = network_predictions(network, X_test)

stats = classification_metrics(y_train, y_pred_train)
print(f"Train: {rounded_dict(stats)}")
stats = classification_metrics(y_val, y_pred_val)
print(f"Val:   {rounded_dict(stats)}")
stats = classification_metrics(y_test, y_pred_test)
print(f"Test:  {rounded_dict(stats)}")

plot_pr_curve(network, X_test, y_test, X_train, y_train)
plot_roc_curve(network, X_test, y_test, X_train, y_train)
