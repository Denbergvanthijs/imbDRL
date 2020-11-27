from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from imbDRL.data import get_train_test_val, load_creditcard
from imbDRL.metrics import classification_metrics
from imbDRL.utils import rounded_dict
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.metrics import (AUC, FalseNegatives, FalsePositives,
                                      Precision, Recall)
from tensorflow.keras.optimizers import Adam

imb_rate = 0.001729  # Imbalance rate
min_class = [1]  # Minority classes
maj_class = [0]  # Majority classes
X_train, y_train, X_test, y_test, = load_creditcard(normalization=True)
X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_val(X_train, y_train, X_test, y_test, min_class, maj_class, val_frac=0.2)
tensorboard_callback = TensorBoard(log_dir=f"./logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")

model = Sequential([Dense(256, activation="relu", input_shape=(X_train.shape[-1],)),
                    Dropout(0.2),
                    Dense(256, activation="relu"),
                    Dropout(0.2),
                    Dense(1, activation="sigmoid")])

metrics = [FalseNegatives(name="FN"),
           FalsePositives(name="FP"),
           Precision(name="Precision"),
           Recall(name="Recall"),
           AUC(curve="ROC", name="AUROC"),
           AUC(curve="PR", name="AUPRC")]

model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=metrics)

model.fit(X_train,
          y_train,
          epochs=30,
          batch_size=2048,
          validation_data=(X_val, y_val),
          callbacks=[tensorboard_callback])

y_pred_val = model(X_val).numpy()
y_pred_test = model(X_test).numpy()
stats = classification_metrics(y_test, np.around(y_pred_test).astype(int))
print(f"Baseline: {rounded_dict(stats)}")

# Validation F1 of every threshold
thresholds = np.arange(0, 1, 0.01)
f1scores = [classification_metrics(y_val, (y_pred_val >= th).astype(int)).get("F1") for th in thresholds]

# Select threshold with highest validation F1
dta_stats = classification_metrics(y_test, (y_pred_test >= thresholds[np.argmax(f1scores)]).astype(int))

print(f"Best DTA Threshold={thresholds[np.argmax(f1scores)]}, F-Score={max(f1scores)}")
print(f"DTA: {rounded_dict(dta_stats)}")
plt.plot(thresholds, f1scores)
plt.xlabel("Thresholds")
plt.ylabel("F1")
plt.xlim((-0.05, 1.05))
plt.xlim((-0.05, 1.05))
plt.show()

# Baseline: {'Gmean': 0.891986, 'F1': 0.795918, 'Precision': 0.795918, 'Recall': 0.795918, 'TP': 78, 'TN': 56844, 'FP': 20, 'FN': 20}
# DTA: {'Gmean': 0.897685, 'F1': 0.80203, 'Precision': 0.79798, 'Recall': 0.806122, 'TP': 79, 'TN': 56844, 'FP': 20, 'FN': 19}
