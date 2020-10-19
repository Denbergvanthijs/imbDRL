import dqnimp.data as data
import numpy as np
import pytest


def test_load_image():
    """Tests dqnimp.data.load_image."""
    # Empty `data_source`
    with pytest.raises(ValueError) as exc:
        data.load_image("")
    assert "No valid" in str(exc.value)

    # Integer `data_source`
    with pytest.raises(ValueError) as exc:
        data.load_image(1234)
    assert "No valid" in str(exc.value)

    # Non-existing `data_source`
    with pytest.raises(ValueError) as exc:
        data.load_image("credit")
    assert "No valid" in str(exc.value)

    image_data = data.load_image("mnist")
    assert [x.shape for x in image_data] == [(60000, 28, 28, 1), (60000, ), (10000, 28, 28, 1), (10000, )]
    assert [x.dtype for x in image_data] == ["float32", "int32", "float32", "int32"]

    image_data = data.load_image("famnist")
    assert [x.shape for x in image_data] == [(60000, 28, 28, 1), (60000, ), (10000, 28, 28, 1), (10000, )]
    assert [x.dtype for x in image_data] == ["float32", "int32", "float32", "int32"]

    image_data = data.load_image("cifar10")
    assert [x.shape for x in image_data] == [(50000, 32, 32, 3), (50000, ), (10000, 32, 32, 3), (10000, )]
    assert [x.dtype for x in image_data] == ["float32", "int32", "float32", "int32"]


def test_load_imdb():
    """Tests dqnimp.data.load_imdb."""
    # Integer `config`
    with pytest.raises(TypeError) as exc:
        data.load_imdb(config=500)
    assert "is no valid datatype" in str(exc.value)

    # Wrong tuple length `config`
    with pytest.raises(ValueError) as exc:
        data.load_imdb(config=(100, 100, 100))
    assert "must be 2" in str(exc.value)

    # Negative `config`
    with pytest.raises(ValueError) as exc:
        data.load_imdb(config=(-100, 10))
    assert "must be > 0" in str(exc.value)

    # Negative `config`
    with pytest.raises(ValueError) as exc:
        data.load_imdb(config=(100, -10))
    assert "must be > 0" in str(exc.value)

    imdb_data = data.load_imdb()
    assert [x.shape for x in imdb_data] == [(25000, 500), (25000, ), (25000, 500), (25000, )]
    assert [x.dtype for x in imdb_data] == ["int32", "int32", "int32", "int32"]


def test_load_creditcard(tmp_path):
    """Tests dqnimp.data.load_creditcard."""
    cols = "Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount,Class\n"
    row = str(list(range(31))).strip("[]") + "\n"

    with pytest.raises(FileNotFoundError) as exc:
        data.load_creditcard(fp_train=tmp_path / "thisfiledoesnotexist.csv")
    assert "fp_train" in str(exc.value)

    with open(data_file := tmp_path / "data_file.csv", "w") as f:
        f.writelines([cols, row, row])

    with pytest.raises(FileNotFoundError) as exc:
        data.load_creditcard(fp_train=data_file, fp_test=tmp_path / "thisfiledoesnotexist.csv")
    assert "fp_test" in str(exc.value)

    with pytest.raises(TypeError) as exc:
        data.load_creditcard(fp_train=data_file, fp_test=data_file, normalization=1234)
    assert "must be of type `bool`" in str(exc.value)

    credit_data = data.load_creditcard(fp_train=data_file, fp_test=data_file)
    assert [x.shape for x in credit_data] == [(2, 29), (2, ), (2, 29), (2, )]
    assert [x.dtype for x in credit_data] == ["float32", "int32", "float32", "int32"]


def test_get_train_test_val():
    """Tests dqnimp.data.get_train_test_val."""
    X = [[1, 2], [3, 4], [5, 6]]
    y = [0, 1, 2]

    with pytest.raises(ValueError) as exc:
        data.get_train_test_val(X, y, X, y, 0.2, [0], [1, 2], val_frac=0.0)
    assert "not in interval" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        data.get_train_test_val(X, y, X, y, 0.2, [0], [1, 2], val_frac=1)
    assert "not in interval" in str(exc.value)

    with pytest.raises(TypeError) as exc:
        data.get_train_test_val(X, y, X, y, 0.2, [0], [1, 2], print_stats=1234)
    assert "must be of type" in str(exc.value)


def test_imbalance_data():
    """Tests dqnimp.data.imbalance_data."""
    X = [7, 7, 7, 8, 8, 8]
    y = [2, 2, 2, 3, 3, 3]

    with pytest.raises(ValueError) as exc:
        data.imbalance_data(X, np.array(y), 0.5, [2], [3])
    assert "`X` must be of type" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        data.imbalance_data(np.array(X), y, 0.5, [2], [3])
    assert "`y` must be of type" in str(exc.value)

    X = np.array(X)
    y = np.array(y)

    with pytest.raises(ValueError) as exc:
        data.imbalance_data(X, y, 0.0, [2], [3])
    assert "not in interval" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        data.imbalance_data(X, y, 1, [2], [3])
    assert "not in interval" in str(exc.value)

    X = np.arange(10)
    y = np.arange(11)

    with pytest.raises(ValueError) as exc:
        data.imbalance_data(X, y, 0.2, [1], [0])
    assert "must contain the same amount of rows" in str(exc.value)

    X = np.arange(100)
    y = np.concatenate([np.ones(50), np.zeros(50)])
    X, y = data.imbalance_data(X, y, 0.2, [1], [0])
    assert [(60, ), (60, ), 10] == [X.shape, y.shape, y.sum()]  # 50/50 is original imb_rate, 10/50(=0.2) is new imb_rate
