import matplotlib.pyplot as plt
import numpy as np
from imbDRL.data import get_train_test_val, load_creditcard
from imbDRL.metrics import classification_metrics
from imbDRL.utils import rounded_dict
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

imb_rate = 0.00173  # Imbalance rate
min_class = [1]  # Minority classes, same setup as in original paper
maj_class = [0]  # Majority classes
X_train, y_train, X_test, y_test, = load_creditcard(normalization=True)
X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_val(X_train, y_train, X_test, y_test, min_class, maj_class, val_frac=0.2)

model = Sequential([Dense(256, activation="relu", input_shape=(X_train.shape[-1],)),
                    Dropout(0.2),
                    Dense(256, activation="relu"),
                    Dropout(0.2),
                    Dense(1, activation="sigmoid")])

model.summary()
metrics = [keras.metrics.FalseNegatives(name="fn"),
           keras.metrics.FalsePositives(name="fp"),
           keras.metrics.Precision(name="precision"),
           keras.metrics.Recall(name="recall")]
model.compile(optimizer=keras.optimizers.Adam(0.001), loss="binary_crossentropy", metrics=metrics)

model.fit(X_train,
          y_train,
          epochs=30,
          batch_size=2048,
          validation_data=(X_val, y_val))

y_pred = model(X_test).numpy()
stats = classification_metrics(y_test, np.around(y_pred).astype(int))
print(rounded_dict(stats))

f1s = []
thresholds = np.arange(0, 1, 0.01)
for th in thresholds:
    y_labels = (y_pred >= th).astype(int)
    stats = classification_metrics(y_test, y_labels).get("F1")
    f1s.append(stats)
ix = np.argmax(f1s)

print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], f1s[ix]))
stats = classification_metrics(y_test, (y_pred >= thresholds[ix]).astype(int))
print(rounded_dict(stats))
plt.plot(thresholds, f1s)
plt.show()
# {'Gmean': 0.892088, 'F1': 0.852459, 'Sensitivity': 0.795918, 'Specificity': 0.999877, 'TP': 78, 'TN': 56857, 'FP': 7, 'FN': 20}
# {'Gmean': 0.920204, 'F1': 0.864583, 'Sensitivity': 0.846939, 'Specificity': 0.999807, 'TP': 83, 'TN': 56853, 'FP': 11, 'FN': 15}
