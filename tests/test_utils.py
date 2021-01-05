import os

import imbDRL.utils as utils
import numpy as np
import pytest


def test_split_csv(tmp_path):
    """Tests imbDRL.utils.split_csv."""
    cols = "V1,V2,Class\n"
    row0 = "0,0,0\n"
    row1 = "1,1,1\n"

    with pytest.raises(FileNotFoundError) as exc:
        utils.split_csv(fp=tmp_path / "thisfiledoesnotexist.csv", fp_dest=tmp_path)
    assert "File at" in str(exc.value)

    data_file = tmp_path / "data_file.csv"
    with open(data_file, "w") as f:
        f.writelines([cols, row0, row0, row1, row1])

    with pytest.raises(ValueError) as exc:
        utils.split_csv(fp=data_file, fp_dest=tmp_path / "thisfolderdoesnotexist")
    assert "Directory at" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        utils.split_csv(fp=data_file, fp_dest=tmp_path, test_size=0.0)
    assert "is not in interval" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        utils.split_csv(fp=data_file, fp_dest=tmp_path, test_size=1)
    assert "is not in interval" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        utils.split_csv(fp=data_file, fp_dest=tmp_path, strat_col="ThisColDoesNotExist")
    assert "not found in DataFrame" in str(exc.value)

    utils.split_csv(fp=data_file, fp_dest=tmp_path, test_size=0.5)
    assert os.path.isfile(tmp_path / "credit0.csv")
    assert os.path.isfile(tmp_path / "credit1.csv")


def test_rounded_dict():
    """Tests imbDRL.utils.rounded_dict."""
    d = {"A": 10.123456789, "B": 100}
    assert utils.rounded_dict(d) == {"A": 10.123457, "B": 100}


def test_imbalance_ratio():
    """Tests imbDRL.utils.imbalance_ratio."""
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert utils.imbalance_ratio(y) == 0.5

    y = np.array([2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
    assert utils.imbalance_ratio(y, 2, 3) == 0.5

    y = np.array(["a", "a", "a", "a", "a", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", ])
    assert utils.imbalance_ratio(y, "a", "b") == 0.5
