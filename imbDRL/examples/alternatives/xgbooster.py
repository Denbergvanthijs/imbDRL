from imbDRL.data import get_train_test_val, load_creditcard
from imbDRL.metrics import classification_metrics
from imbDRL.utils import rounded_dict
from xgboost import XGBClassifier

imb_rate = 0.00173  # Imbalance rate
min_class = [1]  # Minority classes, same setup as in original paper
maj_class = [0]  # Majority classes
X_train, y_train, X_test, y_test, = load_creditcard(normalization=True)
X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_val(X_train, y_train, X_test, y_test, imb_rate, min_class, maj_class)
scale = (y_train.shape[0] - y_train.sum()) // y_train.sum()  # Proportion of majority to minority rows: ~578

model = XGBClassifier(objective="binary:logitraw", scale_pos_weight=scale, eval_metric="aucpr", max_delta_step=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

stats = classification_metrics(y_test, y_pred)
print(rounded_dict(stats))
# ("Gmean", 0.92026 ) ("Fdot5", 0.930493) ("F1", 0.897297) ("F2", 0.866388) ("TP", 83) ("TN", 56860) ("FP", 4) ("FN", 15)
