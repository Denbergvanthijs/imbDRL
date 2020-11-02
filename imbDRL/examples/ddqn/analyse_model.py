from imbDRL.data import get_train_test_val, load_image
from imbDRL.examples.ddqn.example_classes import TrainCustomDDQN
from imbDRL.metrics import (classification_metrics, network_predictions,
                            plot_pr_curve)

imb_rate = 0.01  # Imbalance rate
min_class = [2]  # Minority classes, same setup as in original paper
maj_class = [0, 1, 3, 4, 5, 6, 7, 8, 9]  # Majority classes
fp_model = "./models/20201028_102132"

X_train, y_train, X_test, y_test, = load_image("mnist")
X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_val(X_train, y_train, X_test, y_test, imb_rate, min_class, maj_class)

network = TrainCustomDDQN.load_model(fp_model)
y_pred_val = network_predictions(network, X_val)
y_pred_test = network_predictions(network, X_test)

stats = classification_metrics(y_val, y_pred_val)
print(*[(k, round(v, 6)) for k, v in stats.items()])
stats = classification_metrics(y_test, y_pred_test)
print(*[(k, round(v, 6)) for k, v in stats.items()])

plot_pr_curve(network, X_test, y_test, X_val, y_val)
