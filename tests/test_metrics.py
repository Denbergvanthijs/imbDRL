import imbDRL.metrics as metrics
import numpy as np
import pytest


def test_metrics_by_network():
    """Tests imbDRL.metrics.metrics_by_network."""
    X = [7, 7, 7, 8, 8, 8]
    y = [2, 2, 2, 3, 3, 3]

    with pytest.raises(ValueError) as exc:
        metrics.metrics_by_network([], X, np.array(y))
    assert "`X` must be of type" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        metrics.metrics_by_network([], np.array(X), y)
    assert "`y` must be of type" in str(exc.value)

    X = np.arange(10)
    y = np.arange(11)

    with pytest.raises(ValueError) as exc:
        metrics.metrics_by_network([], X, y)
    assert "must contain the same amount of rows" in str(exc.value)


def test_metrics_by_policy():
    """Tests imbDRL.metrics.metrics_by_policy."""
    X = [7, 7, 7, 8, 8, 8]
    y = [2, 2, 2, 3, 3, 3]

    with pytest.raises(ValueError) as exc:
        metrics.metrics_by_policy([], X, np.array(y))
    assert "`X` must be of type" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        metrics.metrics_by_policy([], np.array(X), y)
    assert "`y` must be of type" in str(exc.value)

    X = np.arange(10)
    y = np.arange(11)

    with pytest.raises(ValueError) as exc:
        metrics.metrics_by_policy([], X, y)
    assert "must contain the same amount of rows" in str(exc.value)


def test_classification_metrics():
    """Tests imbDRL.metrics.classification_metrics."""
    y_true = [7, 7, 7, 8, 8, 8]
    y_pred = [2, 2, 2, 3, 3, 3]

    with pytest.raises(ValueError) as exc:
        metrics.classification_metrics(1, y_pred)
    assert "`y_true` must be of type" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        metrics.classification_metrics(y_true, -1)
    assert "`y_pred` must be of type" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        metrics.classification_metrics(y_true, y_pred + [1])
    assert "must be of same length" in str(exc.value)
