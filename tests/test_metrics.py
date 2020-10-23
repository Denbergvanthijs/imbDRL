from tf_agents.policies.random_tf_policy import RandomTFPolicy
from imbDRL.environments import ClassifyEnv
from tf_agents.environments.tf_py_environment import TFPyEnvironment
import imbDRL.metrics as metrics
import numpy as np
import pytest
import tensorflow as tf


def test_network_predictions():
    """Tests imbDRL.metrics.network_predictions."""
    X = [7, 7, 7, 8, 8, 8]

    with pytest.raises(ValueError) as exc:
        metrics.network_predictions([], X)
    assert "`X` must be of type" in str(exc.value)

    X = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    y_pred = metrics.network_predictions(lambda x: (tf.convert_to_tensor(x), None), X)
    assert np.array_equal(y_pred, [1, 0, 1, 0])


def test_classification_metrics():
    """Tests imbDRL.metrics.classification_metrics."""
    y_true = [1, 1, 1, 1, 1, 1]
    y_pred = [1, 1, 1, 0, 0, 0]

    with pytest.raises(ValueError) as exc:
        metrics.classification_metrics(1, y_pred)
    assert "`y_true` must be of type" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        metrics.classification_metrics(y_true, -1)
    assert "`y_pred` must be of type" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        metrics.classification_metrics(y_true, y_pred + [1])
    assert "must be of same length" in str(exc.value)

    stats = metrics.classification_metrics(y_true, y_pred)
    approx = dict([(k, round(v, 6)) for k, v in stats.items()])
    assert approx == {"Gmean": 0.0, "Fdot5": 0.833333, "F1": 0.666667, "F2": 0.555556, "TP": 3, "TN": 0, "FP": 0, "FN": 3}

    y_true = [1, 1, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 0, 0, 0]
    stats = metrics.classification_metrics(y_true, y_pred)
    approx = dict([(k, round(v, 6)) for k, v in stats.items()])
    assert approx == {"Gmean": 0.0, "Fdot5": 0.0, "F1": 0.0, "F2": 0.0, "TP": 0, "TN": 0, "FP": 0, "FN": 6}

    y_true = [0, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 1, 1, 1]
    stats = metrics.classification_metrics(y_true, y_pred)
    approx = dict([(k, round(v, 6)) for k, v in stats.items()])
    assert approx == {"Gmean": 0.0, "Fdot5": 0.0, "F1": 0.0, "F2": 0.0, "TP": 0, "TN": 0, "FP": 6, "FN": 0}
